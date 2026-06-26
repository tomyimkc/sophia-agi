#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the `storage/` Rust benchmarks on a rented RunPod box with real NVMe.

The sandbox where this repo is developed has only virtio-backed ext4 and limited
cores, so the storage numbers in RESULTS.md (especially the O_DIRECT io_uring
comparison) are honest-but-modest. This tool re-runs them on real datacenter
NVMe + many cores to get credible hardware figures.

It reuses the battle-tested pod lifecycle from tools/runpod_rlvr.py (create pod →
poll SSH → run over SSH → copy report back → ALWAYS delete the pod) and only
swaps the remote payload: install Rust, build the storage workspace, and run the
diskstore (incl. O_DIRECT), kvcache, and infcache benchmarks. We rent the
cheapest available GPU box purely for its NVMe + CPUs; the GPU is unused (a true
CPU-only pod would need a different create payload than the proven GPU one).

    python tools/runpod_storage_bench.py --dry-run            # offline; no pod, no cost
    RUNPOD_API_KEY=... python tools/runpod_storage_bench.py --yes --branch <branch>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runpod_rlvr import (  # noqa: E402 — reuse the proven pod lifecycle
    DEFAULT_REPO_URL,
    PodConnection,
    RunPodError,
    _api_request,
    _build_create_payload,
    _delete_pod,
    _find_pod_by_name,
    _generate_ssh_key,
    _poll_ssh,
    _pod_id,
    _redact,
    _scp_from_pod,
    _ssh_base,
    _stream,
    _wait_ssh_login,
)

# Cheapest-first: we only need the box's NVMe + cores, not the GPU. Community
# cloud is cheapest; pass --cloud-type SECURE for more consistent NVMe.
DEFAULT_GPU_TYPES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 3090",
]

# A plain Ubuntu image is enough (we apt-install build-essential + curl, then
# rustup). The pytorch base also works but is heavier; default to ubuntu.
DEFAULT_IMAGE = "ubuntu:22.04"


def _remote_bench_script(args: argparse.Namespace) -> str:
    branch_flag = f" --branch {shlex.quote(args.branch)}" if args.branch else ""
    repo = shlex.quote(args.repo_url)
    # NOTE: keep this script free of literal { } (no awk, no ${VAR}) so the
    # f-string needs no brace-escaping.
    return f"""set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mkdir -p /workspace/sophia-runpod
cd /workspace/sophia-runpod

echo '[bench] installing toolchain (build-essential, curl, git)…'
if ! command -v cc >/dev/null || ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  apt-get update -y && apt-get install -y --no-install-recommends build-essential curl git ca-certificates pkg-config
fi
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
. "$HOME/.cargo/env"
rustc --version

if [ ! -d sophia-agi/.git ]; then
  git clone --depth 1{branch_flag} {repo} sophia-agi
fi
cd sophia-agi
(git rev-parse HEAD || true) | tee /workspace/sophia-runpod/repo-head.txt
cd storage

REPORT=/workspace/sophia-runpod/storage_bench_report.md
: > "$REPORT"
say() {{ echo "$@" | tee -a "$REPORT"; }}

say "# Sophia storage benchmarks — RunPod"
say ""
say "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say ""
say '## Host'
say '```'
say "kernel : $(uname -r)"
lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core' | tee -a "$REPORT" || true
say "memory : $(grep MemTotal /proc/meminfo)"
say "fs type: $(stat -f -c %T .)"
say "block devices:"
lsblk -d -o NAME,ROTA,SIZE,MODEL 2>/dev/null | tee -a "$REPORT" || true
say "(ROTA=0 means non-rotational / SSD-NVMe)"
say '```'
say ""

say '## Correctness (cargo test)'
say '```'
cargo test --release --quiet 2>&1 | grep -E 'test result|error' | tee -a "$REPORT" || true
cargo test --release --quiet -p diskstore --features io_uring 2>&1 | grep -E 'test result|error' | tee -a "$REPORT" || true
say '```'
say ""

echo '[bench] building release binaries…'
cargo build --release --quiet 2>&1 | tail -3 || true
cargo build --release --quiet -p diskstore --features io_uring 2>&1 | tail -3 || true

run_bench() {{
  # $1 = section title; remaining args = command
  local title="$1"; shift
  say "## $title"
  say '```'
  ( "$@" ) 2>&1 | tee -a "$REPORT" || say '(bench failed — see SSH log)'
  say '```'
  say ""
}}

run_bench "diskstore — O_DIRECT cold I/O (pread vs io_uring) — the real NVMe test" \\
  ./target/release/diskstore-odirect-bench --blocks {args.odirect_blocks} --reads {args.odirect_reads} --depth {args.depth}

run_bench "diskstore — page-cached batched reads (pread vs io_uring)" \\
  ./target/release/diskstore-bench --keys {args.ds_keys} --value-size 512 --batch 256 --batches {args.ds_batches}

run_bench "kvcache — sharded async, no pipelining" \\
  ./target/release/kvcache-bench --clients {args.kv_clients} --ops {args.kv_ops} --pipeline 1

run_bench "kvcache — sharded async, pipeline depth 16" \\
  ./target/release/kvcache-bench --clients {args.kv_clients} --ops {args.kv_ops} --pipeline 16

run_bench "infcache — prefix-cache token reuse (shared system prompt)" \\
  ./target/release/infcache-bench --requests {args.inf_requests} --system 4096 --suffix 128

echo '===== storage_bench_report.md ====='
cat "$REPORT"
echo '[bench] storage benchmark complete.'
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--yes", action="store_true", help="actually create a paid pod (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="print payload + remote script; no pod, no cost")
    ap.add_argument("--keep-pod", action="store_true", help="do NOT delete the pod afterwards (debug only)")
    ap.add_argument("--name", default=f"sophia-storage-bench-{timestamp}")
    ap.add_argument("--source", choices=["git"], default="git")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--branch", default="", help="git branch/tag to clone (the storage feature branch)")
    # bench sizing (kept modest so a run finishes in a few minutes / a few cents)
    ap.add_argument("--odirect-blocks", type=int, default=500_000, help="4 KiB blocks for the O_DIRECT file (~2 GiB)")
    ap.add_argument("--odirect-reads", type=int, default=200_000)
    ap.add_argument("--depth", type=int, default=128)
    ap.add_argument("--ds-keys", type=int, default=500_000)
    ap.add_argument("--ds-batches", type=int, default=4_000)
    ap.add_argument("--kv-clients", type=int, default=64)
    ap.add_argument("--kv-ops", type=int, default=50_000)
    ap.add_argument("--inf-requests", type=int, default=3_000)
    # pod sizing
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES), help="comma-separated GPU type preference (box rented for its NVMe; GPU unused)")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="COMMUNITY")
    ap.add_argument("--interruptible", action="store_true", help="cheaper spot pod")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=20)
    ap.add_argument("--volume-gb", type=int, default=20)
    ap.add_argument("--allowed-cuda-versions", default="")
    ap.add_argument("--no-remote-delete-watchdog", action="store_true",
                    help="disable the on-pod self-delete watchdog (the local 'finally' still deletes)")
    ap.add_argument("--ssh-timeout-s", type=int, default=1200)
    ap.add_argument("--auto-exit-seconds", type=int, default=60 * 60, help="pod self-deletes after this many seconds as a cost backstop")
    ap.add_argument("--artifacts-dir", type=Path,
                    default=ROOT / "agi-proof" / "benchmark-results" / "runpod-storage")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()

    # --dry-run is fully offline and needs no ssh tooling or key.
    if args.dry_run:
        payload = _build_create_payload(args, "ssh-ed25519 <dry-run-placeholder>", api_key=api_key)
        sanitized = json.loads(json.dumps(payload))
        sanitized["env"]["PUBLIC_KEY"] = "ssh-ed25519 …"
        if "RUNPOD_API_KEY" in sanitized.get("env", {}):
            sanitized["env"]["RUNPOD_API_KEY"] = _redact(api_key)
        print(f"[runpod] api key env={args.api_key_env}, value={_redact(api_key)}")
        print("[runpod] create payload (sanitized):")
        print(json.dumps(sanitized, indent=2))
        print("[runpod] remote bench script:")
        print(_remote_bench_script(args))
        print("[runpod] dry-run only; no pod created")
        return 0

    for tool in ("ssh", "scp", "ssh-keygen"):
        if not shutil.which(tool):
            raise RunPodError(f"{tool} not found on PATH (run from CI or a machine with an ssh client)")

    with tempfile.TemporaryDirectory(prefix="sophia-storage-bench-") as tmp:
        tmpdir = Path(tmp)
        key_path, public_key = _generate_ssh_key(tmpdir)
        payload = _build_create_payload(args, public_key, api_key=api_key)
        sanitized = json.loads(json.dumps(payload))
        sanitized["env"]["PUBLIC_KEY"] = "ssh-ed25519 …"
        print("[runpod] create payload (sanitized):")
        print(json.dumps(sanitized, indent=2))

        if not args.yes:
            raise RunPodError("Refusing to create a paid pod without --yes. Use --dry-run to inspect first.")
        if not api_key:
            raise RunPodError(f"Set {args.api_key_env}=<RunPod API key> before running.")

        pod_id = ""
        exit_code = 1
        try:
            try:
                pod = _api_request("POST", "/pods", api_key, payload)
            except RunPodError as exc:
                print(f"[runpod] create errored; scanning for orphan named {args.name!r}: {exc}")
                pod = _find_pod_by_name(api_key, args.name)
                if not pod:
                    raise
            pod_id = _pod_id(pod)
            print(f"[runpod] created pod {pod_id}; costPerHr={pod.get('costPerHr')}")
            conn: PodConnection = _poll_ssh(api_key, pod_id, timeout_s=args.ssh_timeout_s)
            print(f"[runpod] SSH mapped: root@{conn.public_ip} -p {conn.ssh_port}")
            _wait_ssh_login(conn, key_path)

            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_dir / f"{pod_id}.bench.log"
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=_remote_bench_script(args))
            print(f"[runpod] remote command exit code: {exit_code}; log={log_path}")

            _scp_from_pod(conn, key_path, "/workspace/sophia-runpod/storage_bench_report.md",
                          args.artifacts_dir / f"{pod_id}.storage_bench_report.md")
            _scp_from_pod(conn, key_path, "/workspace/sophia-runpod/repo-head.txt",
                          args.artifacts_dir / f"{pod_id}.repo-head.txt")
            return exit_code
        finally:
            if pod_id and not args.keep_pod:
                _delete_pod(api_key, pod_id)
            elif pod_id:
                print(f"[runpod] --keep-pod set; pod still running (costs accrue): {pod_id}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"[runpod] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
