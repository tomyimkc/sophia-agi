#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run a REAL gate-disciplined LoRA training pipeline on a rented RunPod CUDA GPU.

Unlike tools/runpod_speedup.py (a timing micro-benchmark), this trains an actual adapter
and measures real quality through the gate, end-to-end on one pod:

  prepare_lora_dataset  ->  train_lora (--4bit --rslora --neftune --scaffold --guard,
  holdout early-stop)   ->  eval_ladder (base · base+gate · adapter · adapter+gate)
  ->  promote_adapter (W2 protected-floor gate)  ->  copy adapter + reports back.

It reuses the proven RunPod lifecycle from tools/runpod_rlvr.py (create pod -> poll SSH ->
run over SSH -> copy artifacts -> ALWAYS delete pod). Default is --dry-run (no pod, no cost).

    python tools/runpod_train.py --dry-run
    RUNPOD_API_KEY=... python tools/runpod_train.py --yes --branch <branch> --epochs 1
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

DEFAULT_GPU_TYPES = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]
# torch-2.8 base so requirements-lora (pinned <2.9) stays ABI-stable; see runpod_speedup.py.
DEFAULT_TRAIN_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"
ADAPTER_DIR = "/workspace/sophia-runpod/sophia-agi/training/lora/checkpoints/sophia-cuda-v1"


def _remote_train_script(args: argparse.Namespace) -> str:
    branch_flag = (" --branch " + shlex.quote(args.branch)) if args.branch else ""
    adapter_dir = shlex.quote(args.adapter_dir)
    train_data = getattr(args, "train_data", None)
    train_only = getattr(args, "train_only", False)

    if train_data:
        prepare_step = f"# 1) sealed curriculum pack (no prepare_lora_dataset)\nTRAIN_DATA={shlex.quote(str(train_data))}"
        train_cmd = f"""python tools/train_lora.py \\
  --model "$SOPHIA_MODEL" --train "$TRAIN_DATA" --4bit \\
  --epochs "$SOPHIA_EPOCHS" --seed "$SOPHIA_SEED" \\
  --output {adapter_dir}"""
    else:
        prepare_step = "# 1) data (decontaminated train/holdout + pre-split)\npython tools/prepare_lora_dataset.py"
        train_cmd = f"""python tools/train_lora.py \\
  --model "$SOPHIA_MODEL" --4bit --rslora --neftune-alpha 5 --weight-decay 0.05 \\
  --scaffold --guard --eval-every 25 --patience 4 \\
  --epochs "$SOPHIA_EPOCHS" --seed "$SOPHIA_SEED" \\
  --output {adapter_dir}"""

    eval_block = ""
    if not train_only:
        eval_block = f"""
# 3) eval ladder: base · base+gate · adapter · adapter+gate (writes eval_ladder_adapter.json)
python tools/eval_ladder.py --backend hf --model "$SOPHIA_MODEL" --adapter {adapter_dir} \\
  || echo "[train] eval_ladder failed (non-fatal); adapter still returned"
cp training/local_sophia_v2/eval_ladder_adapter.json /workspace/sophia-runpod/eval_ladder_adapter.json 2>/dev/null || true

# 4) W2 promotion gate (protected-floor proof; reads the eval ladder + adapter seed)
python tools/promote_adapter.py \\
  --adapter-config {adapter_dir}/sophia_lora_config.json \\
  --out /workspace/sophia-runpod/promotion.public-report.json \\
  || echo "[train] promote_adapter failed (non-fatal)"

echo "===== sophia_lora_config.json ====="; cat /workspace/sophia-runpod/sophia_lora_config.json 2>/dev/null || true
echo "===== eval_ladder_adapter.json ====="; cat /workspace/sophia-runpod/eval_ladder_adapter.json 2>/dev/null || true
echo "===== promotion.public-report.json ====="; cat /workspace/sophia-runpod/promotion.public-report.json 2>/dev/null || true"""

    return f"""
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export PIP_CACHE_DIR=/workspace/.cache/pip
export SOPHIA_MODEL={shlex.quote(args.model)}
export SOPHIA_EPOCHS={shlex.quote(str(args.epochs))}
export SOPHIA_SEED={shlex.quote(str(args.seed))}
mkdir -p /workspace/sophia-runpod /workspace/.cache/huggingface/hub /workspace/.cache/pip
cd /workspace/sophia-runpod
if [ {shlex.quote(args.source)} = "git" ] && [ ! -d sophia-agi/.git ]; then
  git clone --depth 1{branch_flag} {shlex.quote(args.repo_url)} sophia-agi
fi
cd sophia-agi
(git rev-parse HEAD || true) | tee /workspace/sophia-runpod/repo-head.txt
nvidia-smi || true
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-lora.txt   # torch pinned <2.9 -> ABI stable

{prepare_step}

# 2) REAL training — adapter is tarred immediately after so it always comes back
#    even if eval/promote later fail.
{train_cmd}
if [ -d {adapter_dir} ]; then
  tar -czf /workspace/sophia-runpod/sophia-cuda-v1.tar.gz -C $(dirname {adapter_dir}) $(basename {adapter_dir})
  cp {adapter_dir}/sophia_lora_config.json /workspace/sophia-runpod/sophia_lora_config.json || true
fi
{eval_block}
echo "Sophia real training run complete."
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--yes", action="store_true", help="actually create a RunPod pod (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="print payload + remote script; no pod, no cost")
    ap.add_argument("--keep-pod", action="store_true", help="do NOT delete the pod after the run (debug only)")
    ap.add_argument("--name", default=f"sophia-train-{timestamp}")
    ap.add_argument("--source", choices=["local", "git"], default="git")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--branch", default="", help="git branch/tag to clone (use the feature branch)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="Training JSONL for SFT-only runs (skips prepare_lora_dataset; uses --train-only recipe)",
    )
    ap.add_argument(
        "--adapter-dir",
        default=ADAPTER_DIR,
        help="Remote adapter output directory inside the pod",
    )
    ap.add_argument(
        "--train-only",
        action="store_true",
        help="Skip eval_ladder and promote_adapter on the pod (curriculum / sealed-pack SFT)",
    )
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--interruptible", action="store_true", help="use cheaper spot/interruptible pod")
    ap.add_argument("--image-name", default=DEFAULT_TRAIN_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=80)
    ap.add_argument("--volume-gb", type=int, default=40)
    ap.add_argument("--allowed-cuda-versions", default="")
    ap.add_argument("--no-remote-delete-watchdog", action="store_true")
    ap.add_argument("--ssh-timeout-s", type=int, default=600,
                    help="seconds to wait for SSH mapping PER attempt (lower so a flake retries sooner)")
    ap.add_argument("--ssh-attempts", type=int, default=3,
                    help="recreate the pod up to N times if it never maps SSH (RunPod provisioning flake)")
    ap.add_argument("--auto-exit-seconds", type=int, default=3 * 60 * 60)
    ap.add_argument("--artifacts-dir", type=Path,
                    default=ROOT / "agi-proof" / "benchmark-results" / "runpod-train")
    return ap.parse_args(argv)


def _create_pod_with_ssh(api_key, payload, name, *, attempts, ssh_timeout_s):
    """Create a pod and wait for SSH; on RunPod's intermittent 'RUNNING but no public
    IP/ports' provisioning flake, DELETE the unreachable pod and retry (up to `attempts`).
    Returns (pod_id, conn) for the first reachable pod; raises if every attempt fails.
    Each failed pod is deleted before the next attempt, so no pod is left billing."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        pod_id = ""
        try:
            try:
                pod = _api_request("POST", "/pods", api_key, payload)
            except RunPodError as exc:
                print(f"[runpod] create errored; scanning for orphan named {name!r}: {exc}")
                pod = _find_pod_by_name(api_key, name)
                if not pod:
                    raise
                print(f"[runpod] recovered created pod after error: {pod.get('id')}")
            pod_id = _pod_id(pod)
            print(f"[runpod] attempt {attempt}/{attempts}: created pod {pod_id}; "
                  f"costPerHr={pod.get('costPerHr')}, gpu={pod.get('gpu')}")
            conn = _poll_ssh(api_key, pod_id, timeout_s=ssh_timeout_s)
            print(f"[runpod] SSH mapped: root@{conn.public_ip} -p {conn.ssh_port}")
            return pod_id, conn
        except RunPodError as exc:
            last_exc = exc
            tail = "; deleting unreachable pod and retrying" if attempt < attempts else ""
            print(f"[runpod] attempt {attempt}/{attempts} failed to map SSH ({exc}){tail}")
            if pod_id:
                _delete_pod(api_key, pod_id)
    raise RunPodError(f"no SSH-reachable pod after {attempts} attempt(s): {last_exc}")


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    if args.train_data and not args.train_only:
        args.train_only = True
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()

    if args.dry_run:
        payload = _build_create_payload(args, "ssh-ed25519 <dry-run-placeholder>", api_key=api_key)
        sanitized = json.loads(json.dumps(payload))
        sanitized["env"]["PUBLIC_KEY"] = "ssh-ed25519 …"
        if "RUNPOD_API_KEY" in sanitized["env"]:
            sanitized["env"]["RUNPOD_API_KEY"] = _redact(api_key)
        print(f"[runpod] api key env={args.api_key_env}, value={_redact(api_key)}")
        print("[runpod] create payload (sanitized):")
        print(json.dumps(sanitized, indent=2))
        print("[runpod] remote training script:")
        print(_remote_train_script(args))
        print("[runpod] dry-run only; no pod created")
        return 0

    for tool in ("ssh", "scp", "ssh-keygen"):
        if not shutil.which(tool):
            raise RunPodError(f"{tool} not found on PATH")

    with tempfile.TemporaryDirectory(prefix="sophia-train-") as tmp:
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
        print("[runpod] remote training script:")
        print(_remote_train_script(args))

        if not args.yes:
            raise RunPodError("Refusing to create a paid pod without --yes. Use --dry-run to inspect first.")
        if not api_key:
            raise RunPodError(f"Set {args.api_key_env}=<RunPod API key> before running.")

        pod_id = ""
        conn: PodConnection | None = None
        exit_code = 1
        try:
            pod_id, conn = _create_pod_with_ssh(
                api_key, payload, args.name,
                attempts=args.ssh_attempts, ssh_timeout_s=args.ssh_timeout_s,
            )
            _wait_ssh_login(conn, key_path)
            if args.source == "local":
                _rsync_repo_to_pod(conn, key_path)

            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_dir / f"{pod_id}.train.log"
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=_remote_train_script(args))
            print(f"[runpod] remote command exit code: {exit_code}; log={log_path}")

            for remote, local in (
                ("sophia-cuda-v1.tar.gz", f"{pod_id}.sophia-cuda-v1.tar.gz"),
                ("sophia_lora_config.json", f"{pod_id}.sophia_lora_config.json"),
                ("eval_ladder_adapter.json", f"{pod_id}.eval_ladder_adapter.json"),
                ("promotion.public-report.json", f"{pod_id}.promotion.public-report.json"),
                ("repo-head.txt", f"{pod_id}.repo-head.txt"),
            ):
                _scp_from_pod(conn, key_path, f"/workspace/sophia-runpod/{remote}", args.artifacts_dir / local)
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
