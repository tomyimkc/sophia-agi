#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the fair-baseline LoRA speedup benchmark on a rented RunPod CUDA GPU.

This is the CUDA counterpart to the Apple-Silicon MLX run: it measures the HONEST
apples-to-apples multiplier (same model / same data / same 1 epoch) across
{fp16-maxpad reference, fp16-dynpad, QLoRA-4bit, Unsloth-4bit} — i.e. experiment #2
AND the padding ablation (experiment #1) from the feasibility doc, neither of which
can run on Apple Silicon.

It reuses the battle-tested RunPod lifecycle from tools/runpod_rlvr.py (create pod →
poll SSH → run over SSH → copy report back → ALWAYS delete pod), and only swaps the
remote payload for the benchmark. Default is --dry-run (no pod, no cost).

    python tools/runpod_speedup.py --dry-run
    RUNPOD_API_KEY=... python tools/runpod_speedup.py --yes --branch <branch> --limit 128
"""

from __future__ import annotations

import argparse
import json
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
    DEFAULT_IMAGE,
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
    _rsync_repo_to_pod,
    _scp_from_pod,
    _ssh_base,
    _startup_cmd,  # noqa: F401 — referenced via _build_create_payload
    _stream,
    _wait_ssh_login,
)

# RTX 4090 first (cheap, the device class the claim targets), then 80GB fallbacks.
DEFAULT_GPU_TYPES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]


def _remote_bench_script(args: argparse.Namespace) -> str:
    branch_flag = (" --branch " + shlex.quote(args.branch)) if args.branch else ""
    return f"""
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip
export SOPHIA_MODEL={shlex.quote(args.model)}
export SOPHIA_LIMIT={shlex.quote(str(args.limit))}
export SOPHIA_SEED={shlex.quote(str(args.seed))}
mkdir -p /workspace/sophia-runpod /workspace/.cache/huggingface /workspace/.cache/pip
cd /workspace/sophia-runpod
if [ {shlex.quote(args.source)} = "git" ] && [ ! -d sophia-agi/.git ]; then
  git clone --depth 1{branch_flag} {shlex.quote(args.repo_url)} sophia-agi
fi
cd sophia-agi
(git rev-parse HEAD || true) | tee /workspace/sophia-runpod/repo-head.txt
nvidia-smi || true
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch precheck failed:", type(exc).__name__, exc)
PY
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lora.txt
# Optional extras; if either install fails, that config is recorded as FAILED by the
# bench and the remaining configs still produce the headline numbers (non-fatal).
python -m pip install flash-attn --no-build-isolation || echo "[bench] flash-attn install failed (non-fatal; --pack config will be skipped)"
python -m pip install "unsloth>=2024.8" || echo "[bench] unsloth install failed (non-fatal)"
python tools/prepare_lora_dataset.py
python tools/bench_lora_speedup.py --limit "$SOPHIA_LIMIT" --model "$SOPHIA_MODEL" --seed "$SOPHIA_SEED"
cp training/lora/bench/speedup_report.json /workspace/sophia-runpod/speedup_report.json || true
echo "===== speedup_report.json ====="
cat /workspace/sophia-runpod/speedup_report.json || true
echo "Sophia speedup benchmark complete."
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--yes", action="store_true", help="actually create a RunPod pod (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="print payload + remote script; no pod, no cost")
    ap.add_argument("--keep-pod", action="store_true", help="do NOT delete the pod after the run (debug only)")
    ap.add_argument("--name", default=f"sophia-speedup-{timestamp}")
    ap.add_argument("--source", choices=["local", "git"], default="git")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--branch", default="", help="git branch/tag to clone (use the feature branch)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--limit", type=int, default=128, help="rows of the subset to time (apples-to-apples)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--interruptible", action="store_true", help="use cheaper spot/interruptible pod")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=80)
    ap.add_argument("--volume-gb", type=int, default=40)
    ap.add_argument("--allowed-cuda-versions", default="")
    ap.add_argument("--no-remote-delete-watchdog", action="store_true")
    ap.add_argument("--ssh-timeout-s", type=int, default=1200)
    ap.add_argument("--auto-exit-seconds", type=int, default=2 * 60 * 60)
    ap.add_argument("--artifacts-dir", type=Path,
                    default=ROOT / "agi-proof" / "benchmark-results" / "runpod-speedup")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()

    # --dry-run is fully offline: it must be inspectable without ssh tooling or a key,
    # so the workflow's sanity-check step never depends on the SSH client being present.
    if args.dry_run:
        payload = _build_create_payload(args, "ssh-ed25519 <dry-run-placeholder>", api_key=api_key)
        sanitized = json.loads(json.dumps(payload))
        sanitized["env"]["PUBLIC_KEY"] = "ssh-ed25519 …"
        if "RUNPOD_API_KEY" in sanitized["env"]:
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
            raise RunPodError(f"{tool} not found on PATH")

    with tempfile.TemporaryDirectory(prefix="sophia-speedup-") as tmp:
        tmpdir = Path(tmp)
        key_path, public_key = _generate_ssh_key(tmpdir)
        payload = _build_create_payload(args, public_key, api_key=api_key)
        sanitized = json.loads(json.dumps(payload))
        sanitized["env"]["PUBLIC_KEY"] = "ssh-ed25519 …"
        if "RUNPOD_API_KEY" in sanitized["env"]:
            sanitized["env"]["RUNPOD_API_KEY"] = _redact(api_key)

        print(f"[runpod] api key env={args.api_key_env}, value={_redact(api_key)}")
        print("[runpod] create payload (sanitized):")
        print(json.dumps(sanitized, indent=2))
        print("[runpod] remote bench script:")
        print(_remote_bench_script(args))

        if not args.yes:
            raise RunPodError("Refusing to create a paid pod without --yes. Use --dry-run to inspect first.")
        if not api_key:
            raise RunPodError(f"Set {args.api_key_env}=<RunPod API key> before running.")

        pod_id = ""
        conn: PodConnection | None = None
        exit_code = 1
        try:
            try:
                pod = _api_request("POST", "/pods", api_key, payload)
            except RunPodError as exc:
                print(f"[runpod] create errored; scanning for orphan named {args.name!r}: {exc}")
                pod = _find_pod_by_name(api_key, args.name)
                if not pod:
                    raise
                print(f"[runpod] recovered created pod after error: {pod.get('id')}")
            pod_id = _pod_id(pod)
            print(f"[runpod] created pod {pod_id}; costPerHr={pod.get('costPerHr')}, gpu={pod.get('gpu')}")
            conn = _poll_ssh(api_key, pod_id, timeout_s=args.ssh_timeout_s)
            print(f"[runpod] SSH mapped: root@{conn.public_ip} -p {conn.ssh_port}")
            _wait_ssh_login(conn, key_path)
            if args.source == "local":
                _rsync_repo_to_pod(conn, key_path)

            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_dir / f"{pod_id}.bench.log"
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=_remote_bench_script(args))
            print(f"[runpod] remote command exit code: {exit_code}; log={log_path}")

            _scp_from_pod(conn, key_path, "/workspace/sophia-runpod/speedup_report.json",
                          args.artifacts_dir / f"{pod_id}.speedup_report.json")
            _scp_from_pod(conn, key_path, "/workspace/sophia-runpod/repo-head.txt",
                          args.artifacts_dir / f"{pod_id}.repo-head.txt")
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
