---
name: wisdom-gpu-prebaked
description: >
  Run Sophia-Wisdom-4B GPU jobs (SFT/ORPO/eval on RunPod) WITHOUT wasting GPU credit. Use
  whenever launching tools/runpod_wisdom_pilot_selfreport.py or the wisdom-pilot-runpod
  workflow, or when a pod restart-loops / dies during setup, or when the user mentions the
  pre-baked image, pip-install death loop, RunPod wastage, or "don't waste resources again".
  Wraps the pre-baked CUDA image (docker/wisdom-pilot) + the mandatory cost-guard runbook.
---

# Wisdom GPU runs without resource wastage

## The failure this prevents (root cause, diagnosed 2026-06-26)
The SSH-free pod runs its whole job inside `dockerStartCmd`. When startup did a multi-minute
`pip install`, RunPod kept **killing the container ~60–90s in**, and RunPod **restarted** it —
a restart loop that re-ran the install and **died again, burning GPU credit** until the job
timeout. There is **no RunPod logs API** to see this after the fact; symptoms are: the run
shows "success" but produces no result, or you see **multiple `pod heartbeat` commits ~45s
apart** on the branch. The M4 ORPO run hit exactly this.

## The standing fix (already in the repo — verify it's present before any run)
1. **Pre-baked image.** `docker/wisdom-pilot/Dockerfile` bakes transformers/peft/trl/accelerate/
   datasets into the image. Built+pushed by `.github/workflows/build-wisdom-image.yml` (a GH
   runner has Docker; the dev exec box does not). → there is NO long pip step to die during.
2. **Import-skip in the pod job.** `tools/runpod_wisdom_pilot_selfreport.py` skips pip entirely
   when the deps already import (pre-baked image) and only falls back to pip on the stock image.
3. **Restart-loop auto-abort.** The launcher's `--wait` watches the pod's `lastStartedAt`; after
   `max_restarts` (3) restarts it **deletes the pod and returns nonzero** instead of looping to
   the timeout.

## MANDATORY runbook (do every time — this is the anti-wastage contract)
1. **Build the image once** (no GPU): dispatch `build-wisdom-image` (workflow_dispatch). When it
   finishes, make the GHCR package **public** (repo → Packages → wisdom-pilot → visibility →
   Public) so RunPod can pull it without registry creds. Skip this step if the image already
   exists and is public.
2. **Always run with the pre-baked image:** dispatch `wisdom-pilot-runpod` with input
   `image = ghcr.io/<owner>/wisdom-pilot:latest` (or pass `--image-name` to the launcher).
3. **Cheap validation FIRST, full run SECOND.** Dispatch `limit=24, runs=1` before any full
   `runs=3` / full-N run. A cheap run surfaces train/eval/ORPO bugs for ~$0.30, not a full pod-hour.
4. **Watch for the restart loop in the first ~6 minutes.** Poll the branch commits; if you see
   **≥2 `pod heartbeat` commits within ~5 min**, it's a restart loop — **cancel the workflow run
   and delete the pod immediately**, don't wait. (The `--wait` auto-abort is the backstop; don't
   rely on it alone.)
5. **ALWAYS end by confirming zero leaked pods:**
   `curl -sS https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"` → expect
   `[]`. Delete anything left from THIS effort (named `sophia-wisdom-pilot-*`). Don't delete other
   efforts' pods without asking.
6. **The Actions "success" is not proof of a result.** It only means the pod was created and is
   gone. Confirm the actual artifact landed on the branch
   (`agi-proof/benchmark-results/wisdom-market/M3-pilot-eval*.json` / `M4-orpo-eval*.json`).

## How a run is wired (reference)
- Workflow: `.github/workflows/wisdom-pilot-runpod.yml` (inputs: confirm, runs, limit, seed,
  mode {sft|orpo}, image). Dispatch against the FEATURE branch; the workflow must also exist on
  `main` for `workflow_dispatch` registration.
- Launcher: `tools/runpod_wisdom_pilot_selfreport.py` (creates pod via RunPod REST, pod runs the
  job + git-pushes result/log via the run's GITHUB_TOKEN + self-deletes; `--wait` keeps the job
  alive and aborts restart loops). On-pod work: `tools/pilot_gemma3_run.py` (SFT) /
  `tools/pilot_gemma3_orpo.py` (ORPO); judge: `tools/judge_pilot_answers.py` (local, OpenRouter).
- Secrets: repo Actions secrets `RUNPOD_API_KEY` + `HF_TOKEN` (gemma-3 is gated).

## If it STILL dies during setup with the pre-baked image
Then it's not pip — check (a) the GHCR package is actually public (RunPod pull-auth failure looks
like an instant death), (b) the gemma-3 weight download (~8GB) isn't filling the volume — raise
`--volume-gb` or pre-cache weights on a RunPod network volume, (c) try a different GPU type /
data center. Do this diagnosis on a `limit=24,runs=1` validation, never a full run.

## Observability: the env probe + log-on-crash (added 2026-06-26, after M4)
A pod that crashes BEFORE writing a result used to leave a STALE log and look like a silent
"success". Two fixes make any failure diagnosable in ONE cheap run:
- `finish()` stages the log FIRST and SEPARATELY (`git add LOG` then result/answers each on its
  own line). The old `git add RESULT ANSWERS LOG` failed as a whole when result/answers didn't
  exist yet, dropping the log too. If a run yields no result, read
  `agi-proof/benchmark-results/runpod-wisdom-pilot/pod-selfreport.log` — it now reflects THIS pod.
- An ENV PROBE is pushed right after the heartbeat (before the heavy build/smoke):
  `agi-proof/benchmark-results/runpod-wisdom-pilot/pod-envprobe.txt` with python/torch/
  transformers/peft/trl/accelerate/datasets versions + a guarded `from trl import ORPOConfig,
  ORPOTrainer` test. It survives even a hard kill. **Poll for this file first** when a run fails.

## DEP VERSIONS MUST BE UPPER-CAPPED (root cause of the 2nd M4 death)
Unbounded floors (`transformers>=4.52`, `trl>=0.12`) resolved to **transformers 5.x + trl 1.7.0**,
and **trl 1.x REMOVED top-level `ORPOConfig`/`ORPOTrainer`** → `ImportError: cannot import name
'ORPOConfig' from 'trl'`. The SFT pilot never caught it (it doesn't import trl). The standing pins
are now `transformers>=4.52,<5`, `trl>=0.12,<0.15`, `peft>=0.13,<1`, `accelerate<2` in BOTH the
Dockerfile and the launcher pip-fallback, and the image BUILD imports `ORPOConfig`/`ORPOTrainer`
so a bad resolution fails the free build, not a GPU pod. If you bump any dep, keep both places in
sync and keep the trl cap below the version that drops ORPO.
