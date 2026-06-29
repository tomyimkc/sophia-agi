#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SSH-LESS launcher for the powered cardinal-virtue GO-path evals (Sophrosyne + Dikaiosyne).

This is the RunPod analogue of the Spark+Mac judge farm: it rents ONE GPU pod big enough to
serve the whole judge/subject stack locally via ollama, runs the exact PR-275 pipeline
(build batteries -> decontam -> 2-family label with kappa gate -> 3-arm real-model eval with
bootstrap CIs), then GIT-PUSHES the public reports + a run log back to the branch and
self-deletes. We create/delete the pod via the RunPod REST API (HTTPS, works from the
egress-limited dev container) and poll the branch for the pushed result.

Why a single big pod with ollama: the eval/label tools talk plain OpenAI-compatible HTTP
(`agent.model.ModelClient`, stdlib urllib — no extra deps). ollama load-on-demand serves
all three roles on one port, distinguished by model name:
  * subject   (no-gate baseline)        default qwen2.5:7b-instruct
  * judge A   (family A, != subject)     default qwen2.5:32b-instruct
  * judge B   (family B, != A, != subj)  default llama3.3:70b-instruct-q4_K_M
That satisfies the spec's "judge != subject, >=2 independent families" with no second box.

This NEVER promotes a result. GO/NO-GO is decided by each tool's pre-registered
`measurement_spec.json` (Delta <= -0.10, 95% CI excluding 0, kappa >= 0.40, guardrails);
a NO-GO is a valid, publishable outcome that stays candidate. `canClaimAGI` stays false.

Secrets read from env: RUNPOD_API_KEY (rent/self-delete the pod) and a GitHub push token
(--gh-token-env, default GH_PILOT_PAT). No HF token is needed (ollama models are public).
None are echoed: the RunPod key is injected into the pod env for self-delete; the git PAT
lives only in the pod's credential store and is scrubbed from the pushed log. Default
--dry-run prints the (redacted) payload + job script only.

    RUNPOD_API_KEY=... GH_PILOT_PAT=... python tools/runpod_virtue_evals.py --dry-run
    ... --yes      # actually create the paid pod

Cost guard (per the wisdom-gpu-prebaked anti-wastage runbook, adapted in
docs/11-Platform/Cardinal-Virtue-Benchmarks.md §5b): there is NO pip step (stdlib-only tools
+ ollama via curl), so the pip restart-loop can't occur; the one long step is the ~65GB
ollama model pull (120GB volume so it can't fill). Validate the wiring CHEAPLY first with
small distinct models + a tiny label cap, e.g.
    --subject qwen2.5:7b-instruct --judge-a qwen2.5:14b-instruct --judge-b llama3.1:8b --label-limit 24
then run the full default roster (32B + 70B-q4 judges, no --label-limit).
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

from tools.runpod_rlvr import _api_request, RunPodError  # noqa: E402
# Reuse the battle-tested pod lifecycle helpers (leak sweep, fresh-blob completion detect,
# restart-loop guard) verbatim from the wisdom pilot — same proven anti-wastage controls.
from tools.runpod_wisdom_pilot_selfreport import (  # noqa: E402
    _git_blob,
    _sweep_leaked_pods,
    _wait_for_pod_gone,
)

# 70B q4 (~40GB) + 32B (~20GB) co-resident during labeling needs an 80GB card; the 7B
# subject arm is cheap. Prefer A100/H100 80GB; community as a cheaper fallback via --cloud-type.
DEFAULT_GPU_TYPES = ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB", "NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3"]
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"
REPO_SLUG = "tomyimkc/sophia-agi"
NAME_PREFIX = "sophia-virtue-evals"

# The two public reports the pod pushes. The Sophrosyne report is the completion SENTINEL
# (--wait deletes the pod when this lands fresh); both are committed by finish().
SOPHROSYNE_REPORT = "agi-proof/benchmark-results/sophrosyne/sophrosyne-measure-eval.public-report.json"
DIKAIOSYNE_REPORT = "agi-proof/benchmark-results/dikaiosyne/dikaiosyne-justice-eval.public-report.json"
LOG_PATH = "agi-proof/benchmark-results/runpod-virtue-evals/pod-selfreport.log"
HEARTBEAT_PATH = "agi-proof/benchmark-results/runpod-virtue-evals/pod-heartbeat.txt"
ENVPROBE_PATH = "agi-proof/benchmark-results/runpod-virtue-evals/pod-envprobe.txt"


def _job_script(args: argparse.Namespace) -> str:
    """The ENTIRE job, run as the pod's start command (the dev container can't SSH in).

    All model specs point at the in-pod ollama (localhost:11434). Roles are distinguished by
    model name only; ollama loads/unloads on demand so the 70B judge and 7B subject never
    need to be resident at once. Secrets never reach stdout: the PAT lives in the git
    credential store and the log is scrubbed before push."""
    seeds = int(args.seeds)
    subject = args.subject
    judge_a = args.judge_a
    judge_b = args.judge_b
    judge_a_name = args.judge_a_name
    judge_b_name = args.judge_b_name
    base = "http://localhost:11434/v1"
    subject_spec = f"ollama:{subject}@{base}"
    judge_a_spec = f"ollama:{judge_a}@{base}"
    judge_b_spec = f"ollama:{judge_b}@{base}"
    label_limit = f"--limit {int(args.label_limit)}" if args.label_limit else ""
    workers = int(args.workers)
    # NOTE: no `set -x`; secrets must never reach the pushed log. stdout/stderr -> pod.log,
    # which finish() scrubs of the PAT before committing it.
    return f"""
set -Eeuo pipefail
mkdir -p /workspace
exec > >(tee /workspace/pod.log) 2>&1
echo "[pod] cardinal-virtue powered eval job starting $(date -u)"

finish() {{
  code=$?
  set +e
  echo "[pod] job exit code=$code"
  cd /workspace/sophia-agi 2>/dev/null || true
  # Use an alternate sed delimiter so a '/' in the token can't break the substitution
  # (GitHub tokens are slash-free alphanumerics, but '|' is robust regardless).
  sed -i "s|${{GH_PILOT_PAT:-__none__}}|REDACTED|g" /workspace/pod.log 2>/dev/null || true
  if [ -d /workspace/sophia-agi/.git ]; then
    mkdir -p "$(dirname {LOG_PATH})"
    cp /workspace/pod.log {LOG_PATH} 2>/dev/null || true
    # Stage the LOG FIRST and SEPARATELY so an early crash (no reports yet) still pushes a
    # diagnosable log instead of silently dropping it (the wisdom-pilot blind-spot lesson).
    git add {LOG_PATH} 2>/dev/null || true
    git add agi-proof/benchmark-results/sophrosyne/ 2>/dev/null || true
    git add agi-proof/benchmark-results/dikaiosyne/ 2>/dev/null || true
    git add {HEARTBEAT_PATH} {ENVPROBE_PATH} 2>/dev/null || true
    git -c user.email=noreply@anthropic.com -c user.name=Claude commit \
      -m "virtue evals: self-reported result (exit $code) [skip ci]" 2>/dev/null || echo "[pod] nothing to commit"
    for i in 1 2 3 4 5 6 7 8; do
      git pull --rebase -X ours origin {args.branch} >/dev/null 2>&1 || true
      if git push origin HEAD:{args.branch} 2>/tmp/fpush.err; then echo "[pod] RESULT PUSH OK"; break
      else echo "[pod] result push attempt $i failed:"; cat /tmp/fpush.err; sleep $((i*8)); fi
    done
  fi
  # RunPod sets the container hostname to the pod id, so fall back to $(hostname) if
  # RUNPOD_POD_ID is somehow unset — never skip self-delete and leak a billable pod
  # (mirrors tools/runpod_rlvr.py; the launcher's --wait delete-on-result is the backstop).
  POD_ID="${{RUNPOD_POD_ID:-$(hostname)}}"
  if [ -n "${{RUNPOD_API_KEY:-}}" ] && [ -n "$POD_ID" ]; then
    echo "[pod] self-deleting pod $POD_ID"
    curl -fsS --request DELETE --url "https://rest.runpod.io/v1/pods/${{POD_ID}}" \
      --header "Authorization: Bearer $RUNPOD_API_KEY" || true
  fi
  # CRITICAL anti-restart-loop guard: a RunPod pod RE-RUNS its dockerStartCmd whenever the
  # command EXITS — a clean exit would restart-loop the WHOLE judge-farm job, burning GPU.
  # Do NOT exit: hold the container open so the DELETE above (or --wait) tears it down first.
  echo "[pod] job done + result pushed; holding container open to PREVENT a RunPod restart loop"
  sleep 3600 || true
}}
trap finish EXIT

export DEBIAN_FRONTEND=noninteractive
test -n "${{GH_PILOT_PAT:-}}" || {{ echo "FATAL: GH_PILOT_PAT missing in pod env"; exit 3; }}

apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null 2>&1 || true
nvidia-smi || true

# auth git via a credential store (PAT never printed)
git config --global credential.helper store
printf 'https://x-access-token:%s@github.com\\n' "$GH_PILOT_PAT" > /root/.git-credentials
chmod 600 /root/.git-credentials

cd /workspace
# /workspace persists across a container restart; clear any stale clone so a re-run starts clean.
rm -rf sophia-agi
git clone --branch {args.branch} https://github.com/{REPO_SLUG}.git sophia-agi
cd sophia-agi

# EARLY HEARTBEAT push — proves the git-push channel works BEFORE the long judge run, so a
# missing result is diagnosable instead of silent.
mkdir -p "$(dirname {HEARTBEAT_PATH})"
echo "pod ${{RUNPOD_POD_ID:-?}} alive $(date -u) subject={subject} judgeA={judge_a} judgeB={judge_b}" > {HEARTBEAT_PATH}
git add {HEARTBEAT_PATH}
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "virtue evals: pod heartbeat [skip ci]" >/dev/null 2>&1 || true
if git push origin HEAD:{args.branch} 2>/tmp/push.err; then echo "[pod] HEARTBEAT PUSH OK — delivery channel works"; else echo "[pod] HEARTBEAT PUSH FAILED:"; cat /tmp/push.err; fi

# ── ollama: serve subject + both judge families locally (load-on-demand) ──────────────────
curl -fsSL https://ollama.com/install.sh | sh
# keep at most 1 model resident so the 70B judge and 7B subject never co-OOM; the labeler
# alternates families per case so a small keep-alive is fine and bounds VRAM.
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=5m
nohup ollama serve > /workspace/ollama.log 2>&1 &
for i in $(seq 1 60); do curl -fsS {base}/models >/dev/null 2>&1 && break || sleep 2; done
echo "[pod] pulling models (subject + 2 judge families) — this can take a while"
ollama pull {subject}
ollama pull {judge_a}
ollama pull {judge_b}
ollama list || true

# ENV PROBE — pushed IMMEDIATELY so even a hard kill that skips the trap leaves the toolchain
# + served-model list on the branch (turns a pre-run crash into a 1-run diagnosis).
{{ echo "[probe] $(date -u)"; \
   python3 -c "import sys,platform;print('python',sys.version.split()[0],platform.platform())" 2>&1; \
   ollama list 2>&1; \
   python3 -c "from agent.model import resolve_config;print('model client OK', resolve_config('{subject_spec}').model)" 2>&1; \
}} > {ENVPROBE_PATH} 2>&1 || true
git add {ENVPROBE_PATH}
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "virtue evals: pod env probe [skip ci]" >/dev/null 2>&1 || true
git pull --rebase -X ours origin {args.branch} >/dev/null 2>&1 || true
if git push origin HEAD:{args.branch} 2>/tmp/probe.err; then echo "[pod] ENV PROBE PUSH OK"; else echo "[pod] ENV PROBE PUSH FAILED:"; cat /tmp/probe.err; fi

# ── 1. Rebuild + freeze the external batteries (deterministic, byte-identical) ────────────
echo "[pod] building external batteries"
python3 tools/build_sophrosyne_external_battery.py
python3 tools/build_dikaiosyne_external_battery.py

# ── 2. Decontam (pillar 6) — MUST be clean before any scoring ─────────────────────────────
echo "[pod] decontam asserts"
python3 tools/assert_sophrosyne_decontam.py
python3 tools/assert_dikaiosyne_decontam.py

# ── 3. Label with 2 independent judge families (kappa >= 0.40 gate inside each tool) ──────
echo "[pod] labeling Sophrosyne battery (judges: {judge_a_name} / {judge_b_name})"
python3 tools/label_sophrosyne_battery.py \
  --judge-a '{judge_a_spec}' --judge-a-name '{judge_a_name}' \
  --judge-b '{judge_b_spec}' --judge-b-name '{judge_b_name}' \
  --workers {workers} {label_limit}
echo "[pod] labeling Dikaiosyne battery"
python3 tools/label_dikaiosyne_battery.py \
  --judge-a '{judge_a_spec}' --judge-a-name '{judge_a_name}' \
  --judge-b '{judge_b_spec}' --judge-b-name '{judge_b_name}' \
  --workers {workers} {label_limit}

# ── 4. Score 3 arms vs a REAL no-gate baseline (Delta + bootstrap CI over >= {seeds} seeds) ──
echo "[pod] Sophrosyne real-model eval (subject={subject})"
python3 tools/run_sophrosyne_eval.py --model '{subject_spec}' --seeds {seeds} --workers {workers} --write || echo "[pod] sophrosyne eval returned nonzero (NO-GO is a valid outcome)"
echo "[pod] Dikaiosyne real-model eval (subject={subject})"
python3 tools/run_dikaiosyne_eval.py --model '{subject_spec}' --seeds {seeds} --workers {workers} --write || echo "[pod] dikaiosyne eval returned nonzero (NO-GO is a valid outcome)"

echo "[pod] both public reports written; finish() will commit + push them"
"""


def _build_payload(args, env):
    """Build the RunPod pod-create payload. Secrets are NOT handled here — the caller passes
    the ``env`` dict (real values on the launch path, placeholders on dry-run). Keeping secrets
    out of this shared helper means there is no env->this-function->stdout taint path for the
    clear-text-logging scanner to flag when the dry-run prints the result."""
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
        "env": env,
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
    ap.add_argument("--branch", default="claude/virtue-ai-temperance-justice-ngpquf")
    # Model roster (judge != subject, 2 independent families — per the measurement_spec).
    ap.add_argument("--subject", default="qwen2.5:7b-instruct",
                    help="REAL no-gate baseline subject model (ollama tag)")
    ap.add_argument("--judge-a", default="qwen2.5:32b-instruct", help="judge family A (ollama tag)")
    ap.add_argument("--judge-b", default="llama3.3:70b-instruct-q4_K_M",
                    help="judge family B (ollama tag; != A, != subject — the M3 kappa-deflation lesson)")
    ap.add_argument("--judge-a-name", default="qwen")
    ap.add_argument("--judge-b-name", default="llama")
    ap.add_argument("--seeds", type=int, default=3, help="baseline sampling seeds (>=3 per the spec)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent model calls (low keeps VRAM bounded)")
    ap.add_argument("--label-limit", type=int, default=0, help="smoke: label only first N cases/classes (0 = full)")
    ap.add_argument("--registry-auth-id", default=os.environ.get("RUNPOD_REGISTRY_AUTH_ID", ""),
                    help="RunPod container-registry-auth id for pulling a private image")
    ap.add_argument("--name", default=f"{NAME_PREFIX}-{ts}")
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES))
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=120)
    ap.add_argument("--volume-gb", type=int, default=120)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", action="store_true",
                    help="block until the pod self-deletes (keeps a workflow GITHUB_TOKEN valid "
                         "for the pod's push)")
    ap.add_argument("--wait-timeout-min", type=int, default=240)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        # Dry-run shows the payload STRUCTURE only, with LITERAL placeholder creds — it never
        # reads the real secrets, so no secret value can ever reach stdout.
        payload = _build_payload(args, {"RUNPOD_API_KEY": "<redacted>", "GH_PILOT_PAT": "<redacted>"})
        print(json.dumps(payload, indent=2))
        return 0

    api_key = os.environ.get(args.api_key_env, "")
    gh_pat = os.environ.get(args.gh_token_env, "")
    for name, val in ((args.api_key_env, api_key), (args.gh_token_env, gh_pat)):
        if not val:
            raise RunPodError(f"missing env {name}")
    if not args.yes:
        raise RunPodError("Refusing to create a paid pod without --yes (use --dry-run first).")

    # Reap any leaked virtue-eval pods BEFORE creating a new one, so a relaunch never stacks
    # a 2nd paid pod on a runaway.
    _sweep_leaked_pods(api_key, NAME_PREFIX)

    # Baseline the sentinel report blob BEFORE launch so --wait distinguishes THIS run's fresh
    # result from a stale prior one.
    baseline_blob = _git_blob(args.branch, SOPHROSYNE_REPORT)

    # Real launch: inject the live secrets into env here (NOT printed — this payload goes to
    # the RunPod API). RUNPOD_API_KEY lets the pod self-delete; GH_PILOT_PAT lets it push.
    payload = _build_payload(args, {"RUNPOD_API_KEY": api_key, "GH_PILOT_PAT": gh_pat})
    pod = _api_request("POST", "/pods", api_key, payload)
    pod_id = pod.get("id") or pod.get("podId")
    print(json.dumps({"created": True, "podId": pod_id, "costPerHr": pod.get("costPerHr"),
                      "name": args.name, "branch": args.branch,
                      "expectReports": [SOPHROSYNE_REPORT, DIKAIOSYNE_REPORT],
                      "expectLogAt": LOG_PATH, "sentinelReport": SOPHROSYNE_REPORT,
                      "baselineSentinelBlob": baseline_blob}, indent=2))
    print("[virtue-evals] pod is running the judge-farm job autonomously; it will push the two "
          "public reports + log to the branch; the launcher deletes the pod once the fresh "
          "sentinel report lands.")
    if args.wait:
        return _wait_for_pod_gone(api_key, pod_id, args.wait_timeout_min,
                                  branch=args.branch, result_rel=SOPHROSYNE_REPORT,
                                  baseline_blob=baseline_blob)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
