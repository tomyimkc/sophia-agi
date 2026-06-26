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
    _run,
    _scp_from_pod,
    _ssh_base,
    _startup_cmd,  # noqa: F401 — referenced via _build_create_payload
    _stream,
    _wait_ssh_login,
)

# Broad fallback list so concurrent seeds don't fail when one GPU type is out of
# stock — RunPod picks any available type (payload uses gpuTypePriority=availability).
# All listed cards have >=24 GB VRAM, ample for a 7B QLoRA 4-bit run.
DEFAULT_GPU_TYPES = [
    # 24 GB — cheapest, widest availability
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A5000",
    # 48 GB — broad availability fallback
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA L40S",
    "NVIDIA L40",
    "NVIDIA RTX 6000 Ada Generation",
    # 80 GB — last resort (pricier, usually in stock)
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
]
# torch-2.8 base so requirements-lora (pinned <2.9) stays ABI-stable; see runpod_speedup.py.
DEFAULT_TRAIN_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"
ADAPTER_DIR = "/workspace/sophia-runpod/sophia-agi/training/lora/checkpoints/sophia-cuda-v1"


def _scp_to_pod(conn: PodConnection, key_path: Path, local: Path, remote: str) -> None:
    cmd = [
        "scp", "-i", str(key_path), "-P", str(conn.ssh_port),
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        str(local), f"root@{conn.public_ip}:{remote}",
    ]
    proc = _run(cmd, check=False)
    if proc.returncode != 0:
        raise RunPodError(f"scp to pod failed ({local} -> {remote})")


def _seeds_list(args: argparse.Namespace) -> list[int]:
    """Seeds to train on a single pod. ``--seeds 0,1,2`` (on-pod multi-seed) wins;
    otherwise the single ``--seed`` (one seed per pod, the default)."""
    raw = (getattr(args, "seeds", "") or "").strip()
    if raw:
        return [int(s) for s in raw.split(",") if s.strip() != ""]
    return [int(args.seed)]


def _effective_gpu_count(args: argparse.Namespace) -> int:
    """Mode A (parallel on-pod multi-seed) needs one GPU per seed."""
    seeds = _seeds_list(args)
    if len(seeds) > 1 and getattr(args, "on_pod_mode", "parallel") == "parallel":
        return max(args.gpu_count, len(seeds))
    return args.gpu_count


def _hparams_flags(args: argparse.Namespace) -> str:
    """Build the LoRA hyperparameter-override flag string for train_lora.py.

    Returns the flags for whichever of ``--lr / --lora-r / --lora-alpha / --lora-dropout /
    --neftune-alpha / --weight-decay`` were explicitly set on the CLI (others default to
    None and are omitted). Exported as ``$SOPHIA_HPARAMS`` and appended to every train_lora
    invocation, so an autonomous sweep (pretraining/autopilot) can vary these knobs. Empty
    when nothing is overridden → the remote command is byte-identical to the prior behaviour.
    Appended AFTER the recipe's own flags, so argparse last-value-wins lets an override beat a
    hardcoded default (e.g. the full recipe's ``--neftune-alpha 5``).
    """
    overrides = [
        ("--lr", getattr(args, "lr", None)),
        ("--lora-r", getattr(args, "lora_r", None)),
        ("--lora-alpha", getattr(args, "lora_alpha", None)),
        ("--lora-dropout", getattr(args, "lora_dropout", None)),
        ("--neftune-alpha", getattr(args, "neftune_alpha", None)),
        ("--weight-decay", getattr(args, "weight_decay", None)),
    ]
    return " ".join(f"{flag} {val}" for flag, val in overrides if val is not None)


def _seed_train_cmd(args: argparse.Namespace, seed: int, out_dir: str) -> str:
    """The ``train_lora.py`` invocation for one seed → one adapter dir.

    Mirrors the single-seed recipes: sealed-pack minimal (``--train-only`` + a
    pre-built ``--train-data``) vs the full source-discipline recipe.
    """
    if getattr(args, "train_only", False) and args.train_data:
        return (
            'python tools/train_lora.py '
            '--model "$SOPHIA_MODEL" --train ' + shlex.quote(args.train_data) + ' --4bit '
            '--epochs "$SOPHIA_EPOCHS" --seed ' + str(seed) + ' $SOPHIA_HPARAMS --output ' + out_dir
        )
    train_flag = (" --train " + shlex.quote(args.train_data)) if args.train_data else ""
    return (
        'python tools/train_lora.py '
        '--model "$SOPHIA_MODEL" --4bit --rslora --neftune-alpha 5 --weight-decay 0.05 '
        '--scaffold --guard --eval-every 25 --patience 4' + train_flag + ' '
        '--epochs "$SOPHIA_EPOCHS" --seed ' + str(seed) + ' $SOPHIA_HPARAMS --output ' + out_dir
    )


def _remote_multiseed_script(args: argparse.Namespace, seeds: list[int]) -> str:
    """Remote script that trains MULTIPLE seeds on ONE pod.

    - ``--on-pod-mode parallel`` (Mode A): one seed per GPU via
      ``CUDA_VISIBLE_DEVICES``, all backgrounded then ``wait``-ed (needs
      ``gpuCount >= len(seeds)`` — main() bumps it). Wall-clock ≈ one seed.
    - ``--on-pod-mode sequential`` (Mode B): seeds run back-to-back on a single
      GPU. One pod rented once; cheaper than N pods but ≈ N× wall-clock.

    On-pod eval/promote is skipped — each seed's adapter is tarred and returned
    for offline (CI) eval. DPO is not supported in multi-seed mode.
    """
    branch_flag = (" --branch " + shlex.quote(args.branch)) if args.branch else ""
    mode = getattr(args, "on_pod_mode", "parallel")
    base = args.adapter_dir
    data_step = (
        "python tools/build_local_sophia_dataset.py --check\n"
        if args.train_data
        else "python tools/prepare_lora_dataset.py\n"
    )
    train_lines: list[str] = []
    tar_lines: list[str] = []
    for i, seed in enumerate(seeds):
        out_dir = shlex.quote(f"{base}-seed{seed}")
        cmd = _seed_train_cmd(args, seed, out_dir)
        log = f"/workspace/sophia-runpod/train-seed{seed}.log"
        if mode == "parallel":
            train_lines.append(f"CUDA_VISIBLE_DEVICES={i} {cmd} > {log} 2>&1 &")
        else:
            train_lines.append(f"{cmd} 2>&1 | tee {log}")
        tar_lines.append(
            f"if [ -d {out_dir} ]; then "
            f"tar -czf /workspace/sophia-runpod/sophia-cuda-v1-seed{seed}.tar.gz "
            f"-C $(dirname {out_dir}) $(basename {out_dir}); "
            f"cp {out_dir}/sophia_lora_config.json "
            f"/workspace/sophia-runpod/sophia_lora_config-seed{seed}.json || true; fi"
        )
    train_block = "\n".join(train_lines)
    wait_line = "wait" if mode == "parallel" else ""
    tar_block = "\n".join(tar_lines)
    return f"""
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export PIP_CACHE_DIR=/workspace/.cache/pip
export SOPHIA_MODEL={shlex.quote(args.model)}
export SOPHIA_EPOCHS={shlex.quote(str(args.epochs))}
export SOPHIA_HPARAMS={shlex.quote(_hparams_flags(args))}
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

# 1) sealed pack guard / data prep
{data_step}
# 2) on-pod multi-seed training ({mode}); one adapter per seed, no on-pod eval/promote
{train_block}
{wait_line}
# 3) tar each seed's adapter for offline eval
{tar_block}
echo "Sophia multi-seed run complete ({mode}; seeds {','.join(str(s) for s in seeds)})."
"""


def _remote_train_script(args: argparse.Namespace) -> str:
    seeds = _seeds_list(args)
    if len(seeds) > 1:
        return _remote_multiseed_script(args, seeds)
    branch_flag = (" --branch " + shlex.quote(args.branch)) if args.branch else ""
    adapter_dir = shlex.quote(args.adapter_dir)
    train_only = getattr(args, "train_only", False)
    data_step = (
        "# 1) sealed pack guard (holdout never trained)\n"
        "python tools/build_local_sophia_dataset.py --check\n"
    )
    if args.dpo_pairs:
        train_block = f"""
mkdir -p {adapter_dir}
tar -xzf /workspace/sophia-runpod/sft-adapter.tar.gz -C $(dirname {adapter_dir})
python tools/eval_ladder.py --backend hf --model "$SOPHIA_MODEL" --adapter {adapter_dir} \\
  || echo "[dpo] pre-DPO eval_ladder failed (non-fatal)"
cp training/local_sophia_v2/eval_ladder_adapter.json /workspace/sophia-runpod/eval_ladder_sft.json 2>/dev/null || true
python -m pip install -r requirements-rl.txt
python tools/train_dpo.py --model "$SOPHIA_MODEL" --4bit --rslora \\
  --adapter {adapter_dir} --pairs {shlex.quote(args.dpo_pairs)} \\
  --epochs "$SOPHIA_EPOCHS" --seed "$SOPHIA_SEED" --output {adapter_dir}
"""
    elif train_only and args.train_data:
        # Sealed-curriculum SFT (math-code): minimal recipe on a pre-built pack —
        # no source-discipline scaffold/guard, just QLoRA on the verified rows.
        train_block = f"""
python tools/train_lora.py \\
  --model "$SOPHIA_MODEL" --train {shlex.quote(args.train_data)} --4bit \\
  --epochs "$SOPHIA_EPOCHS" --seed "$SOPHIA_SEED" $SOPHIA_HPARAMS \\
  --output {adapter_dir}
"""
    else:
        if args.train_data:
            train_flag = " --train " + shlex.quote(args.train_data)
        else:
            data_step = (
                "# 1) data (decontaminated train/holdout + pre-split)\n"
                "python tools/prepare_lora_dataset.py\n"
            )
            train_flag = ""
        train_block = f"""
python tools/train_lora.py \\
  --model "$SOPHIA_MODEL" --4bit --rslora --neftune-alpha 5 --weight-decay 0.05 \\
  --scaffold --guard --eval-every 25 --patience 4{train_flag} \\
  --epochs "$SOPHIA_EPOCHS" --seed "$SOPHIA_SEED" $SOPHIA_HPARAMS \\
  --output {adapter_dir}
"""
    # Eval ladder + W2 promotion gate run after training UNLESS --train-only
    # (sealed-pack / curriculum SFT just returns the adapter for offline eval).
    eval_block = ""
    if not train_only:
        eval_block = f"""
# 3) eval ladder: base · base+gate · adapter · adapter+gate (writes eval_ladder_adapter.json)
python tools/eval_ladder.py --backend hf --model "$SOPHIA_MODEL" --adapter {adapter_dir} \
  || echo "[train] eval_ladder failed (non-fatal); adapter still returned"
cp training/local_sophia_v2/eval_ladder_adapter.json /workspace/sophia-runpod/eval_ladder_adapter.json 2>/dev/null || true

# 4) W2 promotion gate (protected-floor proof; reads the eval ladder + adapter seed)
python tools/promote_adapter.py \
  --adapter-config {adapter_dir}/sophia_lora_config.json \
  --out /workspace/sophia-runpod/promotion.public-report.json \
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
export SOPHIA_HPARAMS={shlex.quote(_hparams_flags(args))}
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

{data_step}
# 2) training (SFT or DPO-on-SFT). Adapter tarred immediately after.
{train_block}
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
    # LoRA hyperparameter overrides for autonomous sweeps (pretraining/autopilot). Each is
    # None by default -> omitted -> the remote train_lora command is unchanged. When set,
    # they are appended via $SOPHIA_HPARAMS and win over the recipe's hardcoded defaults.
    ap.add_argument("--lr", type=float, default=None, help="override train_lora --lr")
    ap.add_argument("--lora-r", type=int, default=None, help="override LoRA rank")
    ap.add_argument("--lora-alpha", type=int, default=None, help="override LoRA alpha")
    ap.add_argument("--lora-dropout", type=float, default=None, help="override LoRA dropout")
    ap.add_argument("--neftune-alpha", type=float, default=None, help="override NEFTune alpha")
    ap.add_argument("--weight-decay", type=float, default=None, help="override weight decay")
    ap.add_argument("--train-data", default="",
                    help="path to a pre-built sealed train split (e.g. "
                         "training/local_sophia_7b/mlx/train.jsonl or "
                         "training/sophia-math-code-curriculum/sft_all.jsonl). Empty = regenerate "
                         "on-pod via prepare_lora_dataset.py (legacy 3B path).")
    ap.add_argument("--adapter-dir", default=ADAPTER_DIR,
                    help="remote adapter output directory inside the pod "
                         "(per-seed dir for curriculum SFT runs)")
    ap.add_argument("--train-only", action="store_true",
                    help="skip on-pod eval_ladder + promote_adapter (sealed-pack / curriculum SFT; "
                         "adapter is returned for offline eval)")
    ap.add_argument("--seeds", default="",
                    help="comma-separated seeds to train on ONE pod (e.g. '0,1,2'). Empty = single "
                         "--seed, one pod per seed (default). Multi-seed skips on-pod eval/promote "
                         "and returns one adapter tarball per seed.")
    ap.add_argument("--on-pod-mode", choices=["parallel", "sequential"], default="parallel",
                    help="with --seeds: 'parallel' = one seed per GPU (Mode A, needs a multi-GPU "
                         "pod; gpu-count auto-bumped to #seeds); 'sequential' = seeds back-to-back "
                         "on one GPU (Mode B, cheaper, ~N x wall-clock)")
    ap.add_argument("--dpo-pairs", default="",
                    help="if set, run Stage-3 DPO (train_dpo.py) instead of SFT")
    ap.add_argument("--sft-adapter-archive", type=Path, default=None,
                    help="local .tar.gz of Stage-2 SFT adapter (required for --dpo-pairs)")
    ap.add_argument("--ssh-login-timeout-s", type=int, default=600,
                    help="seconds to wait for SSH login after port mapping")
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


def _create_pod_with_ssh(api_key, payload, name, *, attempts, ssh_timeout_s, key_path, login_timeout_s):
    """Create a pod, wait for SSH port mapping AND login; on flake, delete and retry."""
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
            _wait_ssh_login(conn, key_path, timeout_s=login_timeout_s)
            print(f"[runpod] SSH login OK on attempt {attempt}/{attempts}")
            return pod_id, conn
        except RunPodError as exc:
            last_exc = exc
            tail = "; deleting unreachable pod and retrying" if attempt < attempts else ""
            print(f"[runpod] attempt {attempt}/{attempts} failed SSH reachability ({exc}){tail}")
            if pod_id:
                _delete_pod(api_key, pod_id)
    raise RunPodError(f"no SSH-reachable pod after {attempts} attempt(s): {last_exc}")


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    # Mode A (parallel on-pod multi-seed) needs one GPU per seed.
    args.gpu_count = _effective_gpu_count(args)
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
        if args.dpo_pairs and not args.sft_adapter_archive:
            raise RunPodError("--dpo-pairs requires --sft-adapter-archive (Stage-2 SFT tarball).")

        pod_id = ""
        conn: PodConnection | None = None
        exit_code = 1
        try:
            pod_id, conn = _create_pod_with_ssh(
                api_key, payload, args.name,
                attempts=args.ssh_attempts, ssh_timeout_s=args.ssh_timeout_s,
                key_path=key_path, login_timeout_s=args.ssh_login_timeout_s,
            )
            if args.source == "local":
                _rsync_repo_to_pod(conn, key_path)
            if args.sft_adapter_archive:
                _scp_to_pod(
                    conn, key_path, args.sft_adapter_archive,
                    "/workspace/sophia-runpod/sft-adapter.tar.gz",
                )

            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_dir / f"{pod_id}.train.log"
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=_remote_train_script(args))
            print(f"[runpod] remote command exit code: {exit_code}; log={log_path}")

            for remote, local in (
                ("sophia-cuda-v1.tar.gz", f"{pod_id}.sophia-cuda-v1.tar.gz"),
                ("sophia_lora_config.json", f"{pod_id}.sophia_lora_config.json"),
                ("eval_ladder_adapter.json", f"{pod_id}.eval_ladder_adapter.json"),
                ("eval_ladder_sft.json", f"{pod_id}.eval_ladder_sft.json"),
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
