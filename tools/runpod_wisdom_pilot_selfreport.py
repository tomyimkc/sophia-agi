#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SSH-LESS launcher for the M3 pilot: the pod self-reports via git-over-HTTPS.

The dev exec container is HTTPS-egress-only (can't SSH to a pod), so instead of driving
the pod over SSH this puts the ENTIRE job in the pod's start command. The pod (which has
open egress): clones the branch, rebuilds the deterministic gate-passed dataset, LoRA-trains
the language tower of google/gemma-3-4b-it, evaluates base-vs-adapter, then GIT-PUSHES the
eval report + a run log back to the branch and self-deletes. We create/delete the pod via
the RunPod REST API (HTTPS, works from here) and poll the branch for the pushed result.

Secrets read from env: RUNPOD_API_KEY, HF_TOKEN (gemma is gated), and a GitHub push token
(--gh-token-env, default GH_PILOT_PAT). None are echoed: HF_TOKEN/RUNPOD_API_KEY/PAT are
injected into the pod env and the git PAT is stored in a credentials file on the pod, never
printed. Default --dry-run prints the (redacted) payload + job script only.

    RUNPOD_API_KEY=... HF_TOKEN=... GH_PILOT_PAT=... python tools/runpod_wisdom_pilot_selfreport.py --dry-run
    ... --yes      # actually create the pod
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runpod_rlvr import _api_request, _redact, RunPodError  # noqa: E402

DEFAULT_GPU_TYPES = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"]
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"
REPO_SLUG = "tomyimkc/sophia-agi"
RESULT_PATH = "agi-proof/benchmark-results/wisdom-market/M3-pilot-eval.json"
LOG_PATH = "agi-proof/benchmark-results/runpod-wisdom-pilot/pod-selfreport.log"


def _job_script(args: argparse.Namespace) -> str:
    eval_flags = f"--runs {int(args.runs)}" + (f" --limit {int(args.limit)}" if args.limit else "")
    # NOTE: no `set -x`; secrets must never reach the pushed log. The PAT lives only in the
    # git credential store on the ephemeral pod. All stdout/stderr -> /workspace/pod.log,
    # which is scrubbed of the token before being pushed.
    return f"""
set -Eeuo pipefail
mkdir -p /workspace
exec > >(tee /workspace/pod.log) 2>&1
echo "[pod] M3 pilot self-report job starting $(date -u)"

finish() {{
  code=$?
  set +e
  echo "[pod] job exit code=$code"
  cd /workspace/sophia-agi 2>/dev/null || true
  # scrub any accidental token occurrence, then publish log + (if present) result
  sed -i "s/${{GH_PILOT_PAT:-__none__}}/REDACTED/g; s/${{HF_TOKEN:-__none__}}/REDACTED/g" /workspace/pod.log 2>/dev/null || true
  if [ -d /workspace/sophia-agi/.git ]; then
    mkdir -p "$(dirname {LOG_PATH})"
    cp /workspace/pod.log {LOG_PATH} 2>/dev/null || true
    git add {RESULT_PATH} {LOG_PATH} 2>/dev/null || true
    git -c user.email=noreply@anthropic.com -c user.name=Claude commit \
      -m "M3 pilot: self-reported result (exit $code) [skip ci]" 2>/dev/null || echo "[pod] nothing to commit"
    for i in 1 2 3 4 5; do git push origin HEAD:{args.branch} && break || sleep $((i*5)); done
  fi
  if [ -n "${{RUNPOD_API_KEY:-}}" ] && [ -n "${{RUNPOD_POD_ID:-}}" ]; then
    echo "[pod] self-deleting pod $RUNPOD_POD_ID"
    curl -fsS --request DELETE --url "https://rest.runpod.io/v1/pods/${{RUNPOD_POD_ID}}" \
      --header "Authorization: Bearer $RUNPOD_API_KEY" || true
  fi
}}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export HUGGING_FACE_HUB_TOKEN="${{HF_TOKEN:-}}"
test -n "${{HF_TOKEN:-}}" || {{ echo "FATAL: HF_TOKEN missing in pod env"; exit 3; }}
test -n "${{GH_PILOT_PAT:-}}" || {{ echo "FATAL: GH_PILOT_PAT missing in pod env"; exit 3; }}

apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null 2>&1 || true
nvidia-smi || true

# auth git via a credential store (PAT never printed)
git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\\n' "$GH_PILOT_PAT" > /root/.git-credentials
chmod 600 /root/.git-credentials

cd /workspace
git clone --depth 1 --branch {args.branch} https://github.com/{REPO_SLUG}.git sophia-agi
cd sophia-agi

python -m pip install --upgrade pip >/dev/null
python -m pip install -U "transformers>=4.52" "peft>=0.13" accelerate datasets sentencepiece protobuf

# Rebuild the DETERMINISTIC gate-passed dataset (reproducible; no live teacher -> ~730 rows)
python tools/build_sophia_wisdom_dataset.py --stats
wc -l training/local_sophia_v3/mlx/train.jsonl

echo "[pod] SMOKE"
python tools/pilot_gemma3_run.py --smoke

echo "[pod] FULL train + eval ({eval_flags})"
python tools/pilot_gemma3_run.py --train --eval {eval_flags} --out {RESULT_PATH}
echo "[pod] eval written; finish() will commit + push"
"""


def _build_payload(args, api_key, hf_token, gh_pat):
    gpu_types = [g.strip() for g in args.gpu_type.split(",") if g.strip()]
    return {
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
            "HF_TOKEN": hf_token,
            "GH_PILOT_PAT": gh_pat,
        },
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-lc", _job_script(args)],
    }


def parse_args(argv=None):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    ap.add_argument("--gh-token-env", default="GH_PILOT_PAT")
    ap.add_argument("--branch", default="claude/sophia-wisdom-4b-roadmap-jyesip")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--name", default=f"sophia-wisdom-pilot-sr-{ts}")
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=80)
    ap.add_argument("--volume-gb", type=int, default=60)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="block until the pod self-deletes (keeps the caller/job alive so a "
                         "workflow GITHUB_TOKEN stays valid for the pod's push)")
    ap.add_argument("--wait-timeout-min", type=int, default=200)
    return ap.parse_args(argv)


def _wait_for_pod_gone(api_key: str, pod_id: str, timeout_min: int) -> int:
    import time
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        try:
            _api_request("GET", f"/pods/{pod_id}", api_key, timeout=30)
            status = "running"
        except RunPodError as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                print(f"[selfreport] pod {pod_id} is gone (self-deleted after pushing). Done.")
                return 0
            status = f"poll-error: {exc}"
        print(f"[selfreport] pod {pod_id} still {status}; waiting ...", flush=True)
        time.sleep(45)
    print(f"[selfreport] wait timed out after {timeout_min} min; deleting pod {pod_id} to be safe.")
    try:
        _api_request("DELETE", f"/pods/{pod_id}", api_key, timeout=60)
    except RunPodError:
        pass
    return 2


def main(argv=None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get(args.api_key_env, "")
    hf_token = os.environ.get(args.hf_token_env, "")
    gh_pat = os.environ.get(args.gh_token_env, "")

    if args.dry_run:
        payload = _build_payload(args, _redact(api_key), _redact(hf_token), _redact(gh_pat))
        print(json.dumps(payload, indent=2))
        return 0
    for name, val in (("RUNPOD_API_KEY", api_key), ("HF_TOKEN", hf_token), (args.gh_token_env, gh_pat)):
        if not val:
            raise RunPodError(f"missing env {name}")
    if not args.yes:
        raise RunPodError("Refusing to create a paid pod without --yes (use --dry-run first).")

    payload = _build_payload(args, api_key, hf_token, gh_pat)
    pod = _api_request("POST", "/pods", api_key, payload)
    pod_id = pod.get("id") or pod.get("podId")
    print(json.dumps({"created": True, "podId": pod_id, "costPerHr": pod.get("costPerHr"),
                      "name": args.name, "branch": args.branch,
                      "expectResultAt": RESULT_PATH, "expectLogAt": LOG_PATH}, indent=2))
    print("[selfreport] pod is running the job autonomously; it will push the result + log to "
          "the branch and self-delete.")
    if args.wait:
        return _wait_for_pod_gone(api_key, pod_id, args.wait_timeout_min)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
