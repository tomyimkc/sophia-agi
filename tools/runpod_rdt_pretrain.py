#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SSH-LESS launcher for the Recurrent-Depth Transformer (RDT) pretrain — Phase 1.2.

Same SSH-free, self-reporting, cost-guarded pattern as the wisdom pilot
(``runpod_wisdom_pilot_selfreport.py``): the dev/exec box is HTTPS-egress-only, so the WHOLE
job lives in the pod's start command. The pod (open egress) clones the branch, runs the RDT
self-test + ``pretraining.architecture.rdt_pretrain`` on the GPU, GIT-PUSHES the train report
+ log back to the branch, then self-deletes and holds the container open to defeat RunPod's
exit→restart loop. The launcher creates/deletes the pod via the RunPod REST API and, with
``--wait``, deletes the pod the moment a FRESH report lands on the branch (authoritative
completion), with a restart-count fallback — the exact anti-wastage backstops the
``wisdom-gpu-prebaked`` skill documents.

Two modes:
- ``smoke`` (DEFAULT) — the cost-guard "cheap validation FIRST" run: a small config for a few
  hundred steps on the GPU with the hermetic synthetic stream, proving the CUDA training path
  + the spectral-radius invariant + checkpoint round-trip for ~minutes of the cheapest GPU.
  Run this BEFORE any real pretrain.
- ``pretrain`` — a real from-scratch FineWeb-Edu pretrain (configurable). Only after a green smoke.

Secrets from env: RUNPOD_API_KEY and a GitHub push token (``--gh-token-env``, default
GH_PILOT_PAT). No HF_TOKEN needed — FineWeb-Edu and the byte tokenizer are ungated. Nothing is
echoed. ``--dry-run`` (no pod, no cost) is the default-safe path; a paid pod needs ``--yes``.

    RUNPOD_API_KEY=... GH_PILOT_PAT=... python tools/runpod_rdt_pretrain.py --dry-run
    ...                                                                     --yes --wait   # real pod
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runpod_rlvr import _api_request, _redact, RunPodError  # noqa: E402

# Cheap/available GPUs first — a smoke needs no big VRAM; availability-priority picks what's free.
DEFAULT_GPU_TYPES = [
    "NVIDIA RTX 4090", "NVIDIA RTX A5000", "NVIDIA A40",
    "NVIDIA A100 80GB PCIe", "NVIDIA H100 PCIe",
]
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"  # stock image already has torch
REPO_SLUG = "tomyimkc/sophia-agi"
NAME_PREFIX = "sophia-rdt-pretrain"


def _report_path(mode: str) -> str:
    return f"agi-proof/benchmark-results/rdt-pretrain/{mode}-report.json"


def _log_path() -> str:
    return "agi-proof/benchmark-results/rdt-pretrain/pod.log"


def _train_cmd(args: argparse.Namespace, out_dir: str) -> str:
    """The rdt_pretrain invocation per mode. ``--train-args`` overrides the defaults."""
    if args.train_args:
        extra = args.train_args
    elif args.mode == "smoke":
        # Small CUDA config, a few hundred steps, hermetic synthetic data — the cheap gate.
        extra = ("--steps 300 --warmup 30 --batch 16 --seq 256 --vocab 256 "
                 "--d-model 256 --n-heads 8 --n-kv-heads 2 --d-ff 1024 "
                 "--n-prelude 2 --n-coda 2 --n-loop 8 --data-tokens 500000 "
                 "--bf16 --device cuda --dataset synthetic --log-every 50")
    else:  # pretrain — a small from-scratch run; scale up via --train-args once it's green
        extra = ("--steps 20000 --warmup 2000 --batch 32 --seq 1024 --vocab 256 "
                 "--d-model 1024 --n-heads 16 --n-kv-heads 4 --d-ff 4096 "
                 "--n-prelude 4 --n-coda 4 --n-loop 8 --data-tokens 50000000 "
                 "--bf16 --device cuda --dataset fineweb-edu --log-every 100")
    return f"python -m pretraining.architecture.rdt_pretrain {extra} --out {out_dir}"


def _job_script(args: argparse.Namespace) -> str:
    report_path = _report_path(args.mode)
    log_path = _log_path()
    out_dir = "/workspace/rdt-out"
    train_cmd = _train_cmd(args, out_dir)
    need_datasets = "1" if args.mode == "pretrain" and "fineweb" in train_cmd else "0"
    return f"""
set -Eeuo pipefail
mkdir -p /workspace
exec > >(tee /workspace/pod.log) 2>&1
echo "[pod] RDT pretrain self-report job starting $(date -u) mode={args.mode}"

finish() {{
  code=$?
  set +e
  echo "[pod] job exit code=$code"
  cd /workspace/sophia-agi 2>/dev/null || true
  sed -i "s/${{GH_PILOT_PAT:-__none__}}/REDACTED/g" /workspace/pod.log 2>/dev/null || true
  if [ -d /workspace/sophia-agi/.git ]; then
    mkdir -p "$(dirname {log_path})" "$(dirname {report_path})"
    cp /workspace/pod.log {log_path} 2>/dev/null || true
    cp {out_dir}/train-report.json {report_path} 2>/dev/null || true
    git add {log_path} 2>/dev/null || true
    git add {report_path} 2>/dev/null || true
    git -c user.email=noreply@anthropic.com -c user.name=Claude commit \
      -m "RDT pretrain ({args.mode}): self-reported result (exit $code) [skip ci]" 2>/dev/null || echo "[pod] nothing to commit"
    for i in 1 2 3 4 5 6 7 8; do
      git pull --rebase -X ours origin {args.branch} >/dev/null 2>&1 || true
      if git push origin HEAD:{args.branch} 2>/tmp/fpush.err; then echo "[pod] RESULT PUSH OK"; break
      else echo "[pod] result push attempt $i failed:"; cat /tmp/fpush.err; sleep $((i*8)); fi
    done
  fi
  if [ -n "${{RUNPOD_API_KEY:-}}" ] && [ -n "${{RUNPOD_POD_ID:-}}" ]; then
    echo "[pod] self-deleting pod $RUNPOD_POD_ID"
    curl -fsS --request DELETE --url "https://rest.runpod.io/v1/pods/${{RUNPOD_POD_ID}}" \
      --header "Authorization: Bearer $RUNPOD_API_KEY" || true
  fi
  echo "[pod] job done; holding container open to PREVENT a RunPod restart loop"
  sleep 3600 || true
}}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
test -n "${{GH_PILOT_PAT:-}}" || {{ echo "FATAL: GH_PILOT_PAT missing in pod env"; exit 3; }}

apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null 2>&1 || true
nvidia-smi || true

git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\\n' "$GH_PILOT_PAT" > /root/.git-credentials
chmod 600 /root/.git-credentials

cd /workspace
rm -rf sophia-agi
git clone --branch {args.branch} https://github.com/{REPO_SLUG}.git sophia-agi
cd sophia-agi
export PYTHONPATH=/workspace/sophia-agi

# EARLY HEARTBEAT — proves the push channel works BEFORE the GPU job, so a missing result is
# diagnosable, not silent.
mkdir -p agi-proof/benchmark-results/rdt-pretrain
echo "pod ${{RUNPOD_POD_ID:-?}} alive $(date -u) mode={args.mode}" > agi-proof/benchmark-results/rdt-pretrain/pod-heartbeat.txt
git add agi-proof/benchmark-results/rdt-pretrain/pod-heartbeat.txt
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "RDT pretrain: pod heartbeat [skip ci]" >/dev/null 2>&1 || true
if git push origin HEAD:{args.branch} 2>/tmp/push.err; then echo "[pod] HEARTBEAT PUSH OK"; else echo "[pod] HEARTBEAT PUSH FAILED:"; cat /tmp/push.err; fi

# ENV PROBE — torch/cuda versions pushed immediately, survives a hard kill.
{{ echo "[probe] $(date -u)"; \
   python -c "import sys,platform;print('python',sys.version.split()[0],platform.platform())" 2>&1; \
   python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>&1; \
}} > agi-proof/benchmark-results/rdt-pretrain/pod-envprobe.txt 2>&1 || true
git add agi-proof/benchmark-results/rdt-pretrain/pod-envprobe.txt
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "RDT pretrain: pod env probe [skip ci]" >/dev/null 2>&1 || true
git pull --rebase -X ours origin {args.branch} >/dev/null 2>&1 || true
git push origin HEAD:{args.branch} 2>/dev/null || true

# Deps: torch is in the stock image; only the fineweb pretrain needs `datasets`.
if [ "{need_datasets}" = "1" ]; then
  python -c "import datasets" 2>/dev/null || python -m pip install -q datasets
fi

# 1) Architecture self-test on the pod (cheap, GPU-agnostic) — fails fast if the build is bad.
echo "[pod] RDT architecture self-test"
python -m pretraining.architecture.rdt_torch --self-test

# 2) The training run (writes {out_dir}/train-report.json which finish() pushes).
echo "[pod] RDT {args.mode} run"
{train_cmd}
echo "[pod] train report written; finish() will commit + push"
"""


def _build_payload(args, api_key, gh_pat):
    gpu_types = [g.strip() for g in args.gpu_type.split(",") if g.strip()]
    payload = {
        "name": args.name,
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "gpuCount": 1,
        "imageName": args.image_name,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": "/workspace",
        "supportPublicIp": False,
        "interruptible": False,
        "locked": False,
        "env": {
            "RUNPOD_API_KEY": api_key,   # for the pod's self-delete on exit
            "GH_PILOT_PAT": gh_pat,
        },
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-lc", _job_script(args)],
    }
    auth_id = getattr(args, "registry_auth_id", "") or ""
    if auth_id:
        payload["containerRegistryAuthId"] = auth_id
    return payload


def parse_args(argv=None):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--gh-token-env", default="GH_PILOT_PAT")
    ap.add_argument("--branch", default="claude/openmythos-model-training-t095zk")
    ap.add_argument("--mode", choices=["smoke", "pretrain"], default="smoke",
                    help="smoke = cheap GPU validation FIRST (default); pretrain = real run")
    ap.add_argument("--train-args", default="",
                    help="override the rdt_pretrain CLI args for this mode (full passthrough)")
    ap.add_argument("--registry-auth-id", default=os.environ.get("RUNPOD_REGISTRY_AUTH_ID", ""))
    ap.add_argument("--name", default=f"{NAME_PREFIX}-{ts}")
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=40)
    ap.add_argument("--volume-gb", type=int, default=40)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--wait-timeout-min", type=int, default=90)
    return ap.parse_args(argv)


def _git_blob(branch: str, rel_path: str):
    try:
        subprocess.run(["git", "fetch", "-q", "origin", branch], check=False, timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = subprocess.run(["git", "rev-parse", f"origin/{branch}:{rel_path}"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _delete_pod(api_key: str, pod_id: str) -> None:
    try:
        _api_request("DELETE", f"/pods/{pod_id}", api_key, timeout=60)
    except RunPodError:
        pass


def _sweep_leaked_pods(api_key: str, name_prefix: str) -> int:
    try:
        resp = _api_request("GET", "/pods", api_key, timeout=30)
    except RunPodError as exc:
        print(f"[rdt] could not list pods for leak sweep ({exc}); continuing.")
        return 0
    pods = resp.get("pods", resp) if isinstance(resp, dict) else resp
    killed = 0
    for p in (pods or []):
        pid = p.get("id") or p.get("podId")
        nm = p.get("name", "")
        if pid and isinstance(nm, str) and nm.startswith(name_prefix):
            print(f"[rdt] LEAK SWEEP: deleting pre-existing pod {pid} ({nm})")
            _delete_pod(api_key, pid)
            killed += 1
    return killed


def _wait_for_pod_gone(api_key, pod_id, timeout_min, *, branch, result_rel,
                       baseline_blob=None, max_restarts=3) -> int:
    """Delete the pod the moment a FRESH report blob lands on the branch (authoritative), with
    a restart-count fallback — same backstops as the wisdom launcher."""
    deadline = time.time() + timeout_min * 60
    gone_streak = 0
    last_started = None
    restarts = 0
    result_seen_at = None
    while time.time() < deadline:
        if branch and result_rel:
            blob = _git_blob(branch, result_rel)
            if blob and blob != baseline_blob:
                if result_seen_at is None:
                    result_seen_at = time.time()
                    print(f"[rdt] FRESH report on branch ({result_rel}); deleting pod {pod_id}.")
                _delete_pod(api_key, pod_id)
                try:
                    _api_request("GET", f"/pods/{pod_id}", api_key, timeout=30)
                    if time.time() - result_seen_at > 180:
                        print("[rdt] report delivered + delete issued; returning despite slow teardown.")
                        return 0
                except RunPodError:
                    print(f"[rdt] pod {pod_id} gone after report delivered. Done.")
                    return 0
                time.sleep(20)
                continue
        try:
            pod = _api_request("GET", f"/pods/{pod_id}", api_key, timeout=30)
            gone_streak = 0
            started = pod.get("lastStartedAt")
            if last_started is not None and started and started != last_started:
                restarts += 1
                print(f"[rdt] pod {pod_id} RESTARTED ({restarts}/{max_restarts})")
                if restarts >= max_restarts:
                    print(f"[rdt] restart loop — deleting pod {pod_id} to stop GPU wastage.")
                    _delete_pod(api_key, pod_id)
                    return 3
            last_started = started or last_started
            status = "running"
        except RunPodError as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                gone_streak += 1
                if gone_streak >= 4:
                    print(f"[rdt] pod {pod_id} confirmed gone. Done.")
                    return 0
                status = f"transient-404 ({gone_streak}/4)"
            else:
                gone_streak = 0
                status = f"poll-error: {exc}"
        print(f"[rdt] pod {pod_id} still {status}; waiting ...", flush=True)
        time.sleep(45)
    print(f"[rdt] wait timed out after {timeout_min} min; deleting pod {pod_id} to be safe.")
    _delete_pod(api_key, pod_id)
    return 2


def main(argv=None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    gh_pat = os.environ.get(args.gh_token_env, "")

    if args.dry_run:
        payload = _build_payload(args, _redact(api_key), _redact(gh_pat))
        print(json.dumps(payload, indent=2))
        return 0
    for name, val in (("RUNPOD_API_KEY", api_key), (args.gh_token_env, gh_pat)):
        if not val:
            raise RunPodError(f"missing env {name}")
    if not args.yes:
        raise RunPodError("Refusing to create a paid pod without --yes (use --dry-run first).")

    _sweep_leaked_pods(api_key, NAME_PREFIX)
    result_rel = _report_path(args.mode)
    baseline_blob = _git_blob(args.branch, result_rel)

    payload = _build_payload(args, api_key, gh_pat)
    pod = _api_request("POST", "/pods", api_key, payload)
    pod_id = pod.get("id") or pod.get("podId")
    print(json.dumps({"created": True, "podId": pod_id, "costPerHr": pod.get("costPerHr"),
                      "name": args.name, "branch": args.branch, "mode": args.mode,
                      "expectReportAt": result_rel, "expectLogAt": _log_path(),
                      "baselineReportBlob": baseline_blob}, indent=2))
    if args.wait:
        return _wait_for_pod_gone(api_key, pod_id, args.wait_timeout_min,
                                  branch=args.branch, result_rel=result_rel,
                                  baseline_blob=baseline_blob)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
