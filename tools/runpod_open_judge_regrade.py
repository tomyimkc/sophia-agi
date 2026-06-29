#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SSH-less launcher: corroborate a saved result with a SELF-HOSTED OPEN-WEIGHTS judge on RunPod.

This is the metered, GitHub-Actions-only path for open-judge acceptance criterion #2 (Leiden
value 5 — non-proprietary tooling): instead of grading with a proprietary API, a RunPod pod
serves an open-weights model with vLLM's OpenAI-compatible server on localhost and runs
``tools/run_open_judge_regrade.py`` against it, then GIT-PUSHES the receipt to the branch and
self-deletes. We create/delete the pod via the RunPod REST API (HTTPS) and poll the branch for
the pushed receipt — the same self-report pattern as tools/runpod_wisdom_pilot_selfreport.py,
whose battle-tested anti-wastage machinery (restart-loop abort, leaked-pod sweep, self-delete,
result-blob baseline) is reused here.

Secrets from env (never echoed): RUNPOD_API_KEY, a GitHub push token (--gh-token-env, default
GH_PILOT_PAT), and optionally HF_TOKEN (only needed for gated models; Qwen/Llama-open are not).

    RUNPOD_API_KEY=... GH_PILOT_PAT=... python tools/runpod_open_judge_regrade.py --dry-run
    ... --yes --wait      # actually create the pod + block until the receipt lands

Cost discipline (mirror the wisdom-gpu-prebaked runbook): default to a SMALL model
(Qwen2.5-7B-Instruct) on ONE GPU for a cheap first validation; only scale the model/GPU after a
clean cheap run. The re-grade itself is light (~120 short judge calls over the 30-trap set), so
pod cost is dominated by model download + serve time — keep the model small first.
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

# Reuse the proven RunPod plumbing + anti-wastage helpers (no reinvention).
from tools.runpod_rlvr import _api_request, RunPodError  # noqa: E402
from tools.runpod_wisdom_pilot_selfreport import (  # noqa: E402
    REPO_SLUG, _git_blob, _sweep_leaked_pods, _wait_for_pod_gone,
)

# vLLM's official OpenAI-server image (vllm + openai-compatible server + CUDA + python).
# Pinned for reproducible PAID runs (a floating :latest can break mid-spend); override an
# intentional upgrade with --image-name.
DEFAULT_IMAGE = "vllm/vllm-openai:v0.6.6"
# Broad single-GPU set that fits a 7B–32B served model; 70B needs multi-GPU (raise --gpu-count).
DEFAULT_GPU_TYPES = [
    "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB",
    "NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA H100 PCIe",
]
NAME_PREFIX = "sophia-open-judge"
RECEIPT_PATH = "agi-proof/benchmark-results/wisdom-market/open-judge-regrade.json"
LOG_PATH = "agi-proof/benchmark-results/runpod-open-judge/pod-selfreport.log"
HEARTBEAT_PATH = "agi-proof/benchmark-results/runpod-open-judge/pod-heartbeat.txt"


def _job_script(args: argparse.Namespace) -> str:
    """The pod's entire job (dockerStartCmd). Serves vLLM locally, runs the re-grade harness
    against it, pushes the receipt, self-deletes, then holds the container open so a RunPod
    start-command re-run can't restart-loop the job (see the wisdom-gpu-prebaked runbook)."""
    return f"""
set -Eeuo pipefail
mkdir -p /workspace
exec > >(tee /workspace/pod.log) 2>&1
echo "[pod] open-judge re-grade job starting $(date -u)"

finish() {{
  code=$?
  set +e
  echo "[pod] job exit code=$code"
  cd /workspace/sophia-agi 2>/dev/null || true
  sed -i "s/${{GH_PILOT_PAT:-__none__}}/REDACTED/g; s/${{HF_TOKEN:-__none__}}/REDACTED/g" /workspace/pod.log 2>/dev/null || true
  if [ -d /workspace/sophia-agi/.git ]; then
    mkdir -p "$(dirname {LOG_PATH})"
    cp /workspace/pod.log {LOG_PATH} 2>/dev/null || true
    git add {LOG_PATH} 2>/dev/null || true
    git add {RECEIPT_PATH} 2>/dev/null || true
    git -c user.email=noreply@anthropic.com -c user.name=Claude commit \
      -m "open-judge: self-reported re-grade receipt (exit $code) [skip ci]" 2>/dev/null || echo "[pod] nothing to commit"
    for i in 1 2 3 4 5 6 7 8; do
      git pull --rebase -X ours origin "$JOB_BRANCH" >/dev/null 2>&1 || true
      if git push origin HEAD:"$JOB_BRANCH" 2>/tmp/fpush.err; then echo "[pod] RECEIPT PUSH OK"; break
      else echo "[pod] receipt push attempt $i failed:"; cat /tmp/fpush.err; sleep $((i*8)); fi
    done
  fi
  if [ -n "${{RUNPOD_API_KEY:-}}" ] && [ -n "${{RUNPOD_POD_ID:-}}" ]; then
    echo "[pod] self-deleting pod $RUNPOD_POD_ID"
    curl -fsS --request DELETE --url "https://rest.runpod.io/v1/pods/${{RUNPOD_POD_ID}}" \
      --header "Authorization: Bearer $RUNPOD_API_KEY" || true
  fi
  # Anti-restart-loop: a RunPod pod RE-RUNS its dockerStartCmd whenever it EXITS, so never exit
  # cleanly — hold the container open (the DELETE above / launcher --wait tears it down first).
  echo "[pod] done + receipt pushed; holding container open to PREVENT a RunPod restart loop"
  sleep 3600 || true
}}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export HUGGING_FACE_HUB_TOKEN="${{HF_TOKEN:-}}"
test -n "${{GH_PILOT_PAT:-}}" || {{ echo "FATAL: GH_PILOT_PAT missing in pod env"; exit 3; }}

apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null 2>&1 || true
nvidia-smi || true

git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\\n' "$GH_PILOT_PAT" > /root/.git-credentials
chmod 600 /root/.git-credentials

cd /workspace
rm -rf sophia-agi
git clone --branch "$JOB_BRANCH" https://github.com/{REPO_SLUG}.git sophia-agi
cd sophia-agi

# Heartbeat push — proves the git delivery channel works BEFORE the model download/serve, so a
# missing receipt is diagnosable instead of silent.
mkdir -p "$(dirname {HEARTBEAT_PATH})"
echo "pod ${{RUNPOD_POD_ID:-?}} alive $(date -u) model=$JUDGE_MODEL" > {HEARTBEAT_PATH}
git add {HEARTBEAT_PATH}
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "open-judge: pod heartbeat [skip ci]" >/dev/null 2>&1 || true
if git push origin HEAD:"$JOB_BRANCH" 2>/tmp/push.err; then echo "[pod] HEARTBEAT PUSH OK"; else echo "[pod] HEARTBEAT PUSH FAILED:"; cat /tmp/push.err; fi

# Serve the open-weights judge locally (OpenAI-compatible). Bind to localhost only — no inbound
# exposure; the harness calls it on 127.0.0.1.
echo "[pod] starting vLLM OpenAI server for $JUDGE_MODEL"
vllm serve "$JUDGE_MODEL" --host 127.0.0.1 --port 8000 \
  --download-dir /workspace/.cache --max-model-len {int(args.max_model_len)} \
  > /workspace/vllm.log 2>&1 &
VLLM_PID=$!

# Wait until the server is ready (model download can take several minutes); fail-closed on timeout.
ready=""
for i in $(seq 1 {int(args.serve_timeout_s) // 10}); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then ready=1; echo "[pod] vLLM ready after ~$((i*10))s"; break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "[pod] vLLM process died during startup:"; tail -50 /workspace/vllm.log; exit 4; fi
  sleep 10
done
[ -n "$ready" ] || {{ echo "[pod] vLLM did not become ready within {int(args.serve_timeout_s)}s:"; tail -50 /workspace/vllm.log; exit 5; }}

# Run the re-grade against the LOCAL open-weights endpoint (non-proprietary path).
export OPEN_JUDGE_BASE_URL="http://127.0.0.1:8000/v1"
export OPEN_JUDGE_MODEL="$JUDGE_MODEL"
echo "[pod] grading {args.result_path} with the open judge"
python tools/run_open_judge_regrade.py "$RESULT_PATH_IN" --pack "$PACK_PATH_IN" --out {RECEIPT_PATH}
echo "[pod] re-grade complete; receipt written. finish() will commit + push."

# Stop the server so the script reaches its end and the EXIT trap publishes the receipt.
kill "$VLLM_PID" 2>/dev/null || true
"""


def _build_payload(args, api_key, hf_token, gh_pat):
    gpu_types = [g.strip() for g in args.gpu_type.split(",") if g.strip()]
    return {
        "name": args.name,
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "gpuCount": int(args.gpu_count),
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
            # Operator-controlled values pass via env (NOT interpolated into the bash script) so
            # a crafted value with shell metacharacters cannot inject commands inside the paid pod.
            "JUDGE_MODEL": args.model,
            "RESULT_PATH_IN": args.result_path,
            "PACK_PATH_IN": args.pack_path,
            "JOB_BRANCH": args.branch,
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
    ap.add_argument("--branch", default="main")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="open-weights model to serve as the judge (cheap-first default 7B)")
    ap.add_argument("--result-path",
                    default="training/swarm_router/theta_search_2family_result.json",
                    help="saved generation set (must contain raw_generations)")
    ap.add_argument("--pack-path", default="data/search_recall/pack_third_party.jsonl")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--serve-timeout-s", type=int, default=1800,
                    help="max seconds to wait for vLLM (incl. model download) before fail-closed")
    ap.add_argument("--name", default=f"{NAME_PREFIX}-{ts}")
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=60)
    ap.add_argument("--volume-gb", type=int, default=80)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="block until the pod self-deletes (keeps a workflow GITHUB_TOKEN valid "
                         "for the pod's push, and deletes the pod once the fresh receipt lands)")
    ap.add_argument("--wait-timeout-min", type=int, default=60)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        # Preview the request from ARGS ONLY — never construct or print the payload's env block.
        # Its RUNPOD_API_KEY / HF_TOKEN / GH_PILOT_PAT keys are flagged by a clear-text-logging
        # scanner *by field name* even with empty values, and copying the payload taints the copy.
        # The real payload (incl. env + the dockerStartCmd) is built by _build_payload at launch.
        preview = {
            "name": args.name,
            "imageName": args.image_name,
            "cloudType": args.cloud_type,
            "gpuTypeIds": [g.strip() for g in args.gpu_type.split(",") if g.strip()],
            "gpuCount": int(args.gpu_count),
            "containerDiskInGb": args.container_disk_gb,
            "volumeInGb": args.volume_gb,
            "model": args.model,
            "result_path": args.result_path,
            "pack_path": args.pack_path,
            "branch": args.branch,
            "note": ("env (RunPod key, HF token, GitHub PAT + JUDGE_MODEL/RESULT_PATH_IN/"
                     "PACK_PATH_IN/JOB_BRANCH) and the dockerStartCmd are built by "
                     "_build_payload at launch; not shown here."),
        }
        print(json.dumps(preview, indent=2))
        return 0

    api_key = os.environ.get(args.api_key_env, "")
    hf_token = os.environ.get(args.hf_token_env, "")
    gh_pat = os.environ.get(args.gh_token_env, "")
    # HF_TOKEN is optional (only gated models need it); RUNPOD + push token are mandatory.
    for name, val in (("RUNPOD_API_KEY", api_key), (args.gh_token_env, gh_pat)):
        if not val:
            raise RunPodError(f"missing env {name}")
    if not args.yes:
        raise RunPodError("Refusing to create a paid pod without --yes (use --dry-run first).")

    _sweep_leaked_pods(api_key, NAME_PREFIX)
    baseline_blob = _git_blob(args.branch, RECEIPT_PATH)

    payload = _build_payload(args, api_key, hf_token, gh_pat)
    pod = _api_request("POST", "/pods", api_key, payload)
    pod_id = pod.get("id") or pod.get("podId")
    print(json.dumps({"created": True, "podId": pod_id, "costPerHr": pod.get("costPerHr"),
                      "name": args.name, "branch": args.branch, "model": args.model,
                      "expectReceiptAt": RECEIPT_PATH, "expectLogAt": LOG_PATH,
                      "baselineReceiptBlob": baseline_blob}, indent=2))
    print("[open-judge] pod is serving the open judge + running the re-grade; it pushes the "
          "receipt to the branch and the launcher deletes the pod once the fresh receipt lands.")
    if args.wait:
        return _wait_for_pod_gone(api_key, pod_id, args.wait_timeout_min,
                                  branch=args.branch, result_rel=RECEIPT_PATH,
                                  baseline_blob=baseline_blob)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
