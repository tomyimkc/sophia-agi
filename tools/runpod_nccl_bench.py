#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rent one multi-GPU RunPod pod, measure real NCCL all-reduce bandwidth, calibrate the sim.

This is the bridge from the repo's single-pod tooling to a *measured* network model: it
reuses the battle-tested RunPod lifecycle (create pod → poll SSH → rsync repo → run →
copy report back → ALWAYS delete pod) from tools/runpod_rlvr.py, and the on-pod payload is
tools/bench_nccl_allreduce.py under torchrun (one process per GPU). The returned report
feeds tools/calibrate_network_tax.py, replacing the simulator's MODELED NVLink tier with
the pod's MEASURED bus bandwidth.

NVLink bandwidth needs SXM cards, so the GPU preference defaults to multi-GPU SXM types.
Default is --dry-run (no pod, no cost).

    python tools/runpod_nccl_bench.py --dry-run --gpu-count 2
    RUNPOD_API_KEY=... python tools/runpod_nccl_bench.py --yes --gpu-count 2 --branch <branch>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bench_nccl_allreduce import DEFAULT_ITERS, DEFAULT_SIZES_MB, build_torchrun_cmd
from tools.runpod_rlvr import (  # noqa: E402 — reuse the proven pod lifecycle
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
    _rsync_repo_to_pod,
    _scp_from_pod,
    _ssh_base,
    _stream,
    _wait_ssh_login,
)

# Multi-GPU SXM cards so the intra-node fabric is real NVLink/NVSwitch, not PCIe.
DEFAULT_GPU_TYPES = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 SXM",
]
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2204"
REMOTE_REPO = "/workspace/sophia-runpod"
REMOTE_REPORT = "/workspace/nccl-allreduce.public-report.json"


def _remote_bench_script(args: argparse.Namespace) -> str:
    """Bash run on the pod: prepare repo, run the torchrun all-reduce sweep, emit report."""
    torchrun = build_torchrun_cmd(
        args.gpu_count, REMOTE_REPORT,
        sizes_mb=[int(x) for x in args.sizes_mb.split(",")], iters=args.iters,
        script="tools/bench_nccl_allreduce.py",
    )
    if args.source == "git":
        prep = (f"rm -rf {REMOTE_REPO} && git clone --depth 1 "
                + (f"--branch {args.branch} " if args.branch else "")
                + f"{args.repo_url} {REMOTE_REPO}")
    else:
        prep = f"echo 'using rsynced repo at {REMOTE_REPO}'"
    return f"""
set -Eeuo pipefail
{prep}
cd {REMOTE_REPO}
python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())'
nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "[nccl] launching: {torchrun}"
{torchrun}
echo "[nccl] done; report at {REMOTE_REPORT}"
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY", dest="key_env_name")
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--yes", action="store_true", help="actually create a paid pod")
    ap.add_argument("--dry-run", action="store_true", help="print payload + remote script, no pod")
    ap.add_argument("--keep-pod", action="store_true")
    ap.add_argument("--name", default=f"sophia-nccl-{ts}")
    ap.add_argument("--source", choices=["local", "git"], default="local")
    ap.add_argument("--repo-url", default="https://github.com/tomyimkc/sophia-agi.git")
    ap.add_argument("--branch", default="")
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--gpu-count", type=int, default=2, help="GPUs on the single pod (>=2 for all-reduce)")
    ap.add_argument("--sizes-mb", default=",".join(str(s) for s in DEFAULT_SIZES_MB))
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--interruptible", action="store_true")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=60)
    ap.add_argument("--volume-gb", type=int, default=40)
    ap.add_argument("--allowed-cuda-versions", default="")
    ap.add_argument("--ssh-timeout-s", type=int, default=1200)
    ap.add_argument("--auto-exit-seconds", type=int, default=60 * 60)
    ap.add_argument("--no-remote-delete-watchdog", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="after copying the report, run tools/calibrate_network_tax.py on it")
    ap.add_argument("--ssh-endpoint", default="",
                    help="run on an EXISTING pod instead of creating one, e.g. "
                         "root@103.207.149.84:12226 (no API key / no create / no delete)")
    ap.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh" / "id_ed25519",
                    help="private key for --ssh-endpoint")
    ap.add_argument("--artifacts-dir", type=Path,
                    default=ROOT / "agi-proof" / "benchmark-results" / "cluster")
    return ap.parse_args(argv)


def _parse_ssh_endpoint(endpoint: str) -> tuple[str, str, int]:
    """'root@host:port' -> (user, host, port). Defaults: user=root, port=22."""
    user = "root"
    rest = endpoint
    if "@" in rest:
        user, rest = rest.split("@", 1)
    host, _, port = rest.partition(":")
    return user, host, int(port) if port else 22


def run_on_existing_pod(args: argparse.Namespace) -> int:
    """Run the benchmark on a pod the caller already owns (no create/delete, no API key).

    This is the path for "I already have a pod" — point the tool at its SSH endpoint and it
    clones the repo on the pod, runs the torchrun all-reduce sweep, copies the report back,
    and (optionally) recalibrates. Needs a local ssh/scp client with egress to the pod.
    """
    user, host, port = _parse_ssh_endpoint(args.ssh_endpoint)
    conn = PodConnection(pod_id="existing", public_ip=host, ssh_port=port)
    args.source = "git"  # an existing pod has no rsynced repo; clone it
    print(f"[nccl] using existing pod {user}@{host}:{port} (no create/delete)")
    print("[nccl] remote benchmark script:")
    print(_remote_bench_script(args))
    if args.dry_run:
        print("[nccl] dry-run only; nothing executed")
        return 0

    for tool in ("ssh", "scp"):
        if not shutil.which(tool):
            raise RunPodError(f"{tool} not found on PATH (needed for --ssh-endpoint)")
    if not args.ssh_key.exists():
        raise RunPodError(f"ssh key not found: {args.ssh_key}")
    _wait_ssh_login(conn, args.ssh_key)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.artifacts_dir / "existing-pod.nccl.log"
    cmd = _ssh_base(conn, args.ssh_key) + ["bash", "-s"]
    exit_code = _stream(cmd, log_path, input_text=_remote_bench_script(args))
    print(f"[nccl] remote exit code: {exit_code}; log={log_path}")
    local_report = args.artifacts_dir / "nccl-allreduce.public-report.json"
    got = _scp_from_pod(conn, args.ssh_key, REMOTE_REPORT, local_report)
    if got and args.calibrate:
        from tools.calibrate_network_tax import main as calibrate_main
        print("[nccl] calibrating network tax from measured report …")
        calibrate_main(["--from-nccl", str(local_report), "--markdown"])
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu_count < 2:
        raise RunPodError("all-reduce needs --gpu-count >= 2")
    if args.ssh_endpoint:
        return run_on_existing_pod(args)
    api_key = os.environ.get(args.key_env_name, "")
    if not api_key and args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()

    # Display the sanitized payload + remote script first; dry-run needs no SSH tooling.
    # CodeQL taint is object-level: build the display payload from placeholders so the
    # real key never enters the logged object (overwriting a copied field is not enough).
    sanitized = _build_create_payload(args, "ssh-ed25519 …", api_key=_redact(api_key))
    # Log only the key's length (an int) — never any character of the secret itself.
    key_len = len(api_key)
    masked_key = "" if key_len == 0 else ("***" if key_len <= 12 else f"***(len={key_len})")
    print(f"[runpod] api key env={args.key_env_name}, value={masked_key}")
    print("[runpod] create payload (sanitized):")
    print(json.dumps(sanitized, indent=2))
    print("[runpod] remote benchmark script:")
    print(_remote_bench_script(args))

    if args.dry_run:
        print("[runpod] dry-run only; no pod created")
        return 0
    if not args.yes:
        raise RunPodError("Refusing to create a paid pod without --yes. Use --dry-run first.")
    if not api_key:
        raise RunPodError(f"Set {args.key_env_name}=<RunPod API key> before running.")
    for tool in ("ssh", "scp", "ssh-keygen"):
        if not shutil.which(tool):
            raise RunPodError(f"{tool} not found on PATH")

    with tempfile.TemporaryDirectory(prefix="sophia-nccl-") as tmp:
        tmpdir = Path(tmp)
        key_path, public_key = _generate_ssh_key(tmpdir)
        payload = _build_create_payload(args, public_key, api_key=api_key)
        pod_id = ""
        conn: PodConnection | None = None
        try:
            try:
                pod = _api_request("POST", "/pods", api_key, payload)
            except RunPodError as exc:
                print(f"[runpod] create errored; scanning for orphan {args.name!r}: {exc}")
                pod = _find_pod_by_name(api_key, args.name)
                if not pod:
                    raise
            pod_id = _pod_id(pod)
            print(f"[runpod] created pod {pod_id}; gpu={pod.get('gpu')}, costPerHr={pod.get('costPerHr')}")
            conn = _poll_ssh(api_key, pod_id, timeout_s=args.ssh_timeout_s)
            print(f"[runpod] SSH mapped: root@{conn.public_ip} -p {conn.ssh_port}")
            _wait_ssh_login(conn, key_path)
            if args.source == "local":
                _rsync_repo_to_pod(conn, key_path)

            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_dir / f"{pod_id}.nccl.log"
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=_remote_bench_script(args))
            print(f"[runpod] remote exit code: {exit_code}; log={log_path}")

            local_report = args.artifacts_dir / "nccl-allreduce.public-report.json"
            got = _scp_from_pod(conn, key_path, REMOTE_REPORT, local_report)
            if got and args.calibrate:
                from tools.calibrate_network_tax import main as calibrate_main
                print("[nccl] calibrating network tax from measured report …")
                calibrate_main(["--from-nccl", str(local_report), "--markdown"])
            return exit_code
        finally:
            if pod_id and not args.keep_pod:
                _delete_pod(api_key, pod_id)
            elif pod_id:
                print(f"[runpod] --keep-pod set; pod still running: {pod_id}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
