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


def _mode_spec(mode: str):
    """(prefix, on-pod script, extra CLI flags) for a run mode. orpo_sft = the canonical
    SFT->ORPO STACK (train SFT, merge, ORPO on top); orpo = ORPO from base; sft = M3 pilot."""
    if mode == "orpo_sft":
        return "M4-orpo-sft", "pilot_gemma3_orpo.py", "--from-sft"
    if mode == "orpo":
        return "M4-orpo", "pilot_gemma3_orpo.py", ""
    return "M3-pilot", "pilot_gemma3_run.py", ""


def _artifact_paths(args: argparse.Namespace):
    """The branch-relative (result, answers) paths this run will push. Shared by the job script
    and the launcher's --wait so completion is detected by the EXACT file the pod writes."""
    prefix, _, _ = _mode_spec(getattr(args, "mode", "sft"))
    seed = int(args.seed)
    sfx = "" if seed == 0 else f"-seed{seed}"  # seed 0 keeps the canonical name
    return (f"agi-proof/benchmark-results/wisdom-market/{prefix}-eval{sfx}.json",
            f"agi-proof/benchmark-results/wisdom-market/{prefix}-answers{sfx}.json")


def _job_script(args: argparse.Namespace) -> str:
    eval_flags = f"--runs {int(args.runs)}" + (f" --limit {int(args.limit)}" if args.limit else "")
    seed = int(args.seed)
    mode = getattr(args, "mode", "sft")
    prefix, script, mode_flags = _mode_spec(mode)
    result_path, answers_path = _artifact_paths(args)
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
    # Stage the LOG FIRST and SEPARATELY: `git add A B C` fails as a whole if A/B don't
    # exist yet (early crash -> no result/answers), which would silently drop the log too
    # and leave the branch with a STALE log (the M4 blind-spot, diagnosed 2026-06-26).
    git add {LOG_PATH} 2>/dev/null || true
    git add {result_path} 2>/dev/null || true
    git add {answers_path} 2>/dev/null || true
    git -c user.email=noreply@anthropic.com -c user.name=Claude commit \
      -m "M3 pilot: self-reported result (exit $code) [skip ci]" 2>/dev/null || echo "[pod] nothing to commit"
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
  # CRITICAL anti-restart-loop guard (diagnosed 2026-06-26): a RunPod pod RE-RUNS its
  # dockerStartCmd whenever the command EXITS — so a clean exit here restart-loops the WHOLE
  # SFT->ORPO->eval job, burning GPU (~10 cycles observed). Do NOT exit: hold the container
  # open so (a) the DELETE above tears it down first, and (b) if DELETE failed, the launcher's
  # --wait deletes the pod after seeing the pushed result — either way the job NEVER re-runs.
  echo "[pod] job done + result pushed; holding container open to PREVENT a RunPod restart loop"
  sleep 3600 || true
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
# /workspace is a persistent volume; if the container RESTARTS and re-runs this start
# command, a stale clone would make `git clone` fail (exit 128). Clear it first so a
# restart re-runs cleanly instead of dirtying the result with a spurious failure.
rm -rf sophia-agi
git clone --branch {args.branch} https://github.com/{REPO_SLUG}.git sophia-agi
cd sophia-agi

# EARLY HEARTBEAT push — proves the git-push channel (auth + fast-forward) works BEFORE the
# ~50-min job, so a missing result is diagnosable instead of silent. If this fails the whole
# delivery path is broken and the Actions log will show it.
mkdir -p agi-proof/benchmark-results/runpod-wisdom-pilot
echo "pod ${{RUNPOD_POD_ID:-?}} alive $(date -u) seed={seed}" > agi-proof/benchmark-results/runpod-wisdom-pilot/pod-heartbeat.txt
git add agi-proof/benchmark-results/runpod-wisdom-pilot/pod-heartbeat.txt
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "M3 pilot: pod heartbeat (seed {seed}) [skip ci]" >/dev/null 2>&1 || true
if git push origin HEAD:{args.branch} 2>/tmp/push.err; then echo "[pod] HEARTBEAT PUSH OK — delivery channel works"; else echo "[pod] HEARTBEAT PUSH FAILED:"; cat /tmp/push.err; fi

# ENV PROBE — pushed IMMEDIATELY (before the heavy build/smoke) so even a hard kill that
# skips the EXIT trap still leaves the package versions + a guarded TRL/PEFT import test on
# the branch. This is how a pre-model-load crash (e.g. a prebaked-image version conflict)
# becomes diagnosable in ONE run instead of a blind retry.
{{ echo "[probe] $(date -u)"; \
   python -c "import sys,platform;print('python',sys.version.split()[0],platform.platform())" 2>&1; \
   python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1; \
   python -c "import transformers,peft,trl,accelerate,datasets;print('transformers',transformers.__version__,'peft',peft.__version__,'trl',trl.__version__,'accelerate',accelerate.__version__,'datasets',datasets.__version__)" 2>&1; \
   python -c "from trl import ORPOConfig, ORPOTrainer;from peft import LoraConfig;print('trl ORPO import OK')" 2>&1; \
}} > agi-proof/benchmark-results/runpod-wisdom-pilot/pod-envprobe.txt 2>&1 || true
git add agi-proof/benchmark-results/runpod-wisdom-pilot/pod-envprobe.txt
git -c user.email=noreply@anthropic.com -c user.name=Claude commit -m "M4 pilot: pod env probe (seed {seed}) [skip ci]" >/dev/null 2>&1 || true
git pull --rebase -X ours origin {args.branch} >/dev/null 2>&1 || true
if git push origin HEAD:{args.branch} 2>/tmp/probe.err; then echo "[pod] ENV PROBE PUSH OK"; else echo "[pod] ENV PROBE PUSH FAILED:"; cat /tmp/probe.err; fi

# Deps: skip the slow pip install when the image is PRE-BAKED (deps already importable).
# This is the fix for the "pod container dies ~60s into pip install -> restart loop -> GPU
# wastage" failure. With docker/wisdom-pilot (built by build-wisdom-image.yml) imports
# succeed and we skip pip entirely; on the stock image we fall back to pip as before.
if python -c "import transformers,peft,trl,accelerate,datasets,sentencepiece" 2>/dev/null; then
  echo "[pod] deps pre-baked — skipping pip (no long install to die during)"
else
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -U "transformers>=4.52,<5" "peft>=0.13,<1" "trl>=0.12,<0.15" "accelerate<2" datasets sentencepiece protobuf
fi

# Rebuild the DETERMINISTIC gate-passed dataset (reproducible; no live teacher -> ~730 rows)
python tools/build_sophia_wisdom_dataset.py --stats
wc -l training/local_sophia_v3/mlx/train.jsonl

echo "[pod] SMOKE ({mode})"
python tools/{script} --smoke {mode_flags}

echo "[pod] FULL {mode} train + eval ({eval_flags}) seed={seed}"
python tools/{script} --train --eval --seed {seed} {eval_flags} {mode_flags} \
  --out {result_path} --save-answers {answers_path}
echo "[pod] eval + answers written; finish() will commit + push"
"""


def _build_payload(args, api_key, hf_token, gh_pat):
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
            "HF_TOKEN": hf_token,
            "GH_PILOT_PAT": gh_pat,
        },
        "dockerEntrypoint": [],
        "dockerStartCmd": ["bash", "-lc", _job_script(args)],
    }
    # Private-registry pull (e.g. a private GHCR pre-baked image): reference a RunPod
    # container-registry-auth so the image stays private instead of being made public.
    auth_id = getattr(args, "registry_auth_id", "") or ""
    if auth_id:
        payload["containerRegistryAuthId"] = auth_id
    return payload


def parse_args(argv=None):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    ap.add_argument("--gh-token-env", default="GH_PILOT_PAT")
    ap.add_argument("--branch", default="claude/sophia-wisdom-4b-roadmap-jyesip")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["sft", "orpo", "orpo_sft"], default="sft")
    ap.add_argument("--registry-auth-id", default=os.environ.get("RUNPOD_REGISTRY_AUTH_ID", ""),
                    help="RunPod container-registry-auth id for pulling a private image")
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


def _git_blob(branch: str, rel_path: str):
    """Current git blob hash of rel_path on origin/branch, or None if absent. Used to detect
    the pod pushing a FRESH result without trusting self-delete / lastStartedAt."""
    import subprocess
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
    """Delete any pre-existing pods whose name starts with name_prefix BEFORE creating a new
    one. This reaps leaks from earlier runs (e.g. a restart-loop pod whose GH run was cancelled,
    so it can no longer be cleaned up by that run) so a relaunch never stacks a 2nd paid pod on
    a runaway. Safe: runs before this run's own pod exists, and only touches the sophia-wisdom
    pilot name prefix."""
    try:
        resp = _api_request("GET", "/pods", api_key, timeout=30)
    except RunPodError as exc:
        print(f"[selfreport] could not list pods for leak sweep ({exc}); continuing.")
        return 0
    pods = resp.get("pods", resp) if isinstance(resp, dict) else resp
    killed = 0
    for p in (pods or []):
        pid = p.get("id") or p.get("podId")
        nm = p.get("name", "")
        if pid and isinstance(nm, str) and nm.startswith(name_prefix):
            print(f"[selfreport] LEAK SWEEP: deleting pre-existing pod {pid} ({nm})")
            _delete_pod(api_key, pid)
            killed += 1
    if killed:
        print(f"[selfreport] leak sweep removed {killed} stale sophia-wisdom pod(s).")
    return killed


def _wait_for_pod_gone(api_key: str, pod_id: str, timeout_min: int, *, branch: str = "",
                       result_rel: str = "", baseline_blob=None, max_restarts: int = 3) -> int:
    """Wait until the job is DONE, then ensure the pod is deleted. Completion is detected
    AUTHORITATIVELY by the result file landing FRESH on the branch (blob != baseline) — the
    launcher then DELETEs the pod itself rather than trusting the pod's self-delete or a
    `lastStartedAt` restart count. Why: a RunPod pod re-runs its start command on exit WITHOUT
    advancing lastStartedAt, so the old restart-abort never fired and the job restart-looped
    (~10 cycles, GPU waste, 2026-06-26). 404-streak + restart-count remain as fallbacks."""
    import time
    deadline = time.time() + timeout_min * 60
    gone_streak = 0
    last_started = None
    restarts = 0
    result_seen_at = None
    while time.time() < deadline:
        # AUTHORITATIVE completion: the pod pushed a fresh result blob -> delete the pod, done.
        if branch and result_rel:
            blob = _git_blob(branch, result_rel)
            if blob and blob != baseline_blob:
                if result_seen_at is None:
                    result_seen_at = time.time()
                    print(f"[selfreport] FRESH result on branch ({result_rel}); deleting pod {pod_id}.")
                _delete_pod(api_key, pod_id)
                # give the API a moment, then confirm gone
                try:
                    _api_request("GET", f"/pods/{pod_id}", api_key, timeout=30)
                    if time.time() - result_seen_at > 180:
                        print("[selfreport] result delivered + delete issued; returning despite slow teardown.")
                        return 0
                except RunPodError:
                    print(f"[selfreport] pod {pod_id} gone after result delivered. Done.")
                    return 0
                time.sleep(20)
                continue
        try:
            pod = _api_request("GET", f"/pods/{pod_id}", api_key, timeout=30)
            gone_streak = 0
            started = pod.get("lastStartedAt")
            if last_started is not None and started and started != last_started:
                restarts += 1
                print(f"[selfreport] pod {pod_id} RESTARTED ({restarts}/{max_restarts}) — container restarted")
                if restarts >= max_restarts:
                    print(f"[selfreport] restart loop — deleting pod {pod_id} to stop GPU wastage.")
                    _delete_pod(api_key, pod_id)
                    return 3
            last_started = started or last_started
            status = "running"
        except RunPodError as exc:
            if "404" in str(exc) or "not found" in str(exc).lower():
                gone_streak += 1
                if gone_streak >= 4:
                    print(f"[selfreport] pod {pod_id} confirmed gone (4 consecutive 404s). Done.")
                    return 0
                status = f"transient-404 ({gone_streak}/4) — maybe restarting"
            else:
                gone_streak = 0
                status = f"poll-error: {exc}"
        print(f"[selfreport] pod {pod_id} still {status}; waiting ...", flush=True)
        time.sleep(45)
    print(f"[selfreport] wait timed out after {timeout_min} min; deleting pod {pod_id} to be safe.")
    _delete_pod(api_key, pod_id)
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

    # Reap any leaked sophia-wisdom pods (e.g. a restart-loop pod whose GH run was cancelled)
    # BEFORE creating a new one, so a relaunch never stacks a 2nd paid pod on a runaway.
    _sweep_leaked_pods(api_key, "sophia-wisdom-pilot")

    # Baseline the result blob BEFORE launch so --wait can tell THIS run's fresh result from a
    # stale prior one (the file usually already exists on the branch from an earlier run).
    result_rel, _answers_rel = _artifact_paths(args)
    baseline_blob = _git_blob(args.branch, result_rel)

    payload = _build_payload(args, api_key, hf_token, gh_pat)
    pod = _api_request("POST", "/pods", api_key, payload)
    pod_id = pod.get("id") or pod.get("podId")
    print(json.dumps({"created": True, "podId": pod_id, "costPerHr": pod.get("costPerHr"),
                      "name": args.name, "branch": args.branch,
                      "expectResultAt": result_rel, "expectLogAt": LOG_PATH,
                      "baselineResultBlob": baseline_blob}, indent=2))
    print("[selfreport] pod is running the job autonomously; it will push the result + log to "
          "the branch; the launcher deletes the pod once the fresh result lands.")
    if args.wait:
        return _wait_for_pod_gone(api_key, pod_id, args.wait_timeout_min,
                                  branch=args.branch, result_rel=result_rel, baseline_blob=baseline_blob)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
