#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Launch the M3 PILOT (gemma-3-4b LoRA train + base-vs-adapter eval) on a RunPod CUDA pod.

Reuses the PROVEN pod lifecycle from tools/runpod_rlvr.py (create -> poll SSH -> rsync the
working tree -> run over SSH -> scp artifacts back -> ALWAYS delete the pod). The remote
work is tools/pilot_gemma3_run.py. The base is the HF-GATED google/gemma-3-4b-it, so the
HF token is injected via the pod's create-payload env and read on the pod from
/proc/1/environ (it never appears in the streamed log).

Cost discipline: a SMOKE stage (load + 2 train steps + 2-case eval) runs first and ABORTS
the expensive full run if it fails, so a load/version bug costs minutes, not a full pod-hour.
Default is --dry-run (no pod, no cost).

    python tools/runpod_wisdom_pilot.py --dry-run
    RUNPOD_API_KEY=... HF_TOKEN=... python tools/runpod_wisdom_pilot.py --yes --branch <branch>
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

from tools.runpod_rlvr import (  # noqa: E402 — reuse the proven lifecycle
    PodConnection, RunPodError, _build_create_payload, _delete_pod, _generate_ssh_key,
    _redact, _rsync_repo_to_pod, _scp_from_pod, _ssh_base, _stream, _wait_ssh_login,
)
from tools.runpod_train import _create_pod_with_ssh  # noqa: E402

DEFAULT_GPU_TYPES = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"]
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"
REMOTE = "/workspace/sophia-runpod/sophia-agi"


def _remote_script(args: argparse.Namespace) -> str:
    eval_flags = f"--runs {int(args.runs)}" + (f" --limit {int(args.limit)}" if args.limit else "")
    return f"""
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export PIP_CACHE_DIR=/workspace/.cache/pip
# read the gated-model token from PID1 env (injected via create payload) — never logged here
export HF_TOKEN="$(tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p')"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
mkdir -p /workspace/.cache/huggingface/hub /workspace/.cache/pip /workspace/out
cd {REMOTE}
(git rev-parse HEAD 2>/dev/null || echo "rsync-local") | tee /workspace/out/repo-head.txt
nvidia-smi || true
python -m pip install --upgrade pip >/dev/null
python -m pip install -U "transformers>=4.52" "peft>=0.13" accelerate datasets sentencepiece protobuf
test -n "$HF_TOKEN" || {{ echo "FATAL: HF_TOKEN not present in pod env"; exit 3; }}

echo "===== SMOKE (load + 2 steps + 2-case eval) ====="
python tools/pilot_gemma3_run.py --smoke || {{ echo "SMOKE FAILED — aborting before full run"; exit 4; }}
cp agi-proof/benchmark-results/wisdom-market/M3-pilot-smoke.json /workspace/out/ 2>/dev/null || true

echo "===== FULL train + eval ({eval_flags}) ====="
python tools/pilot_gemma3_run.py --train --eval {eval_flags} \
  --out agi-proof/benchmark-results/wisdom-market/M3-pilot-eval.json
cp agi-proof/benchmark-results/wisdom-market/M3-pilot-eval.json /workspace/out/ 2>/dev/null || true
if [ -d training/adapters/sophia-wisdom-4b-pilot ]; then
  tar -czf /workspace/out/adapter.tar.gz -C training/adapters sophia-wisdom-4b-pilot
  cp training/adapters/sophia-wisdom-4b-pilot/pilot_train_meta.json /workspace/out/ 2>/dev/null || true
fi
echo "===== M3-pilot-eval.json ====="; cat /workspace/out/M3-pilot-eval.json 2>/dev/null || true
echo "PILOT REMOTE COMPLETE"
"""


def parse_args(argv=None) -> argparse.Namespace:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    ap.add_argument("--yes", action="store_true", help="actually create a paid pod")
    ap.add_argument("--dry-run", action="store_true", help="print payload + remote script; no pod")
    ap.add_argument("--keep-pod", action="store_true", help="do NOT delete the pod after (debug)")
    ap.add_argument("--name", default=f"sophia-wisdom-pilot-{ts}")
    ap.add_argument("--branch", default="")  # unused with rsync source, kept for parity
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--interruptible", action="store_true")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=80)
    ap.add_argument("--volume-gb", type=int, default=60)
    ap.add_argument("--allowed-cuda-versions", default="")
    ap.add_argument("--no-remote-delete-watchdog", action="store_true")
    ap.add_argument("--ssh-timeout-s", type=int, default=600)
    ap.add_argument("--ssh-attempts", type=int, default=3)
    ap.add_argument("--auto-exit-seconds", type=int, default=4 * 60 * 60)
    ap.add_argument("--artifacts-dir", type=Path,
                    default=ROOT / "agi-proof" / "benchmark-results" / "runpod-wisdom-pilot")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    hf_token = os.environ.get(args.hf_token_env, "")

    if args.dry_run:
        payload = _build_create_payload(args, "ssh-ed25519 <dry-run>", api_key=api_key)
        payload["env"]["HF_TOKEN"] = _redact(hf_token)
        payload["env"]["PUBLIC_KEY"] = "ssh-ed25519 …"
        if "RUNPOD_API_KEY" in payload["env"]:
            payload["env"]["RUNPOD_API_KEY"] = _redact(api_key)
        print("[pilot] create payload (sanitized):")
        print(json.dumps(payload, indent=2))
        print("[pilot] remote script:")
        print(_remote_script(args))
        print("[pilot] dry-run only; no pod created")
        return 0

    for tool in ("ssh", "scp", "ssh-keygen", "rsync"):
        if not shutil.which(tool):
            raise RunPodError(f"{tool} not found on PATH")
    if not args.yes:
        raise RunPodError("Refusing to create a paid pod without --yes. Use --dry-run first.")
    if not api_key:
        raise RunPodError(f"Set {args.api_key_env}=<RunPod API key>.")
    if not hf_token:
        raise RunPodError(f"Set {args.hf_token_env}=<HF token with gemma access>.")

    with tempfile.TemporaryDirectory(prefix="sophia-pilot-") as tmp:
        key_path, public_key = _generate_ssh_key(Path(tmp))
        payload = _build_create_payload(args, public_key, api_key=api_key)
        payload["env"]["HF_TOKEN"] = hf_token  # injected to PID1 env; read on pod from /proc/1/environ

        pod_id = ""
        exit_code = 1
        try:
            pod_id, conn = _create_pod_with_ssh(
                api_key, payload, args.name, attempts=args.ssh_attempts, ssh_timeout_s=args.ssh_timeout_s)
            _wait_ssh_login(conn, key_path)
            _rsync_repo_to_pod(conn, key_path)
            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_dir / f"{pod_id}.pilot.log"
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=_remote_script(args))
            print(f"[pilot] remote exit code: {exit_code}; log={log_path}")
            for remote, local in (
                ("M3-pilot-eval.json", f"{pod_id}.M3-pilot-eval.json"),
                ("M3-pilot-smoke.json", f"{pod_id}.M3-pilot-smoke.json"),
                ("pilot_train_meta.json", f"{pod_id}.pilot_train_meta.json"),
                ("adapter.tar.gz", f"{pod_id}.adapter.tar.gz"),
                ("repo-head.txt", f"{pod_id}.repo-head.txt"),
            ):
                try:
                    _scp_from_pod(conn, key_path, f"/workspace/out/{remote}", args.artifacts_dir / local)
                except Exception as exc:
                    print(f"[pilot] scp {remote} skipped: {exc}")
            return exit_code
        finally:
            if pod_id and not args.keep_pod:
                _delete_pod(api_key, pod_id)
            elif pod_id:
                print(f"[pilot] --keep-pod set; pod still running: {pod_id}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
