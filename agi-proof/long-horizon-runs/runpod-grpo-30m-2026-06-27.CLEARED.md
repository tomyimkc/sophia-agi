# long_horizon_30m — CLEARED by a genuine live GRPO run (2026-06-27)

**Status:** the `long_horizon_30m` Level-3 blocker (the 1800s `short-30min` tier in
`tools/run_long_horizon.py::classify_tier`) is **CLEARED** by a real GPU training run,
not by padding. `canClaimAGI` stays **False** — this clears the autonomy-duration tier,
it is not an AGI claim.

## The run
- **Harness report:** `runpod-grpo-30m-2026-06-27.public-report.json`
  - `durationSec` = **2489.72** → `tier` = **`short-30min`** (≥1800s)
  - `objectivePassed` = **true**; `autonomy.level` = **full-autonomy**; `humanInterventionCount` = 0
- **What the harness timed:** `tools/run_long_horizon.py` ran **on the GitHub runner**
  (which has SSH egress to the rented pod; the CPU dev box does not — see
  `HONEST-WALL-2026-06-27.md` §4), wrapping the real `tools/runpod_rlvr.py` live GRPO
  launch as its long step. So the harness's **own monotonic clock** measured real GPU
  compute end-to-end, not a sleep loop.
- **The real compute (pod `5d62vnc013olna`, on-demand A100 80GB):** live GRPO on
  `zai-org/glm-4-9b-chat-hf` via vLLM colocate (trl 0.19.1 / vllm 0.9.1 /
  transformers 4.53.2), **113 optimizer steps**, 1 epoch, seed 0, provenance reward.
  Real training curves (reward ~0.51→0.79 across steps, KL rising, LR decaying);
  held-out eval over 94 cases; trained LoRA adapter tarred + sha256'd and copied back.
  Train log: `agi-proof/benchmark-results/runpod-rlvr/5d62vnc013olna.train.log` (1245 lines).
- **Pod teardown:** `runpod_rlvr.py` deleted the pod in its `finally` block —
  `[runpod] pod 5d62vnc013olna terminated` (recorded in the harness JSONL tool_call).
  A remote watchdog (`--auto-exit-seconds`) is the backstop. (RunPod MCP could not be
  used to double-check from this box — its configured key returns 401 — so teardown is
  evidenced from the run log, not an independent pods listing.)
- **Provenance:** GitHub Actions run
  `https://github.com/tomyimkc/sophia-agi/actions/runs/28277509269`; workflow
  `.github/workflows/long-horizon-runpod.yml` (merged to main via PR #192 with required
  checks `validate-core`+`test` green); branch `claude/sophia-agi-long-horizon-30m-qhlc3t`.

## This run's actual RLVR numbers (honest scope — NOT a validated capability claim)
From the fresh pod eval `5d62vnc013olna.rlvr.adapter-eval.json`:
- meanReward (the SSIL "capability" metric): **0.5819 → 0.7404** (Δ +0.1585)
- passAt1: **0.5106 → 0.5319** (Δ +0.021)
- integrity 0.8696 → 0.8696 (unchanged); contamination-free; `adapterImprovesMeanReward`=true
- SSIL Layer-1 (re-ingested fresh, see below): `verdict=promote`, hardened
  `combinedVerdict=promote` with only GUARD+GOOD enforced and 10 gates `pending`.
  **Boundary:** single run, no CI, no ≥2 judge families, no κ — this is a candidate/rung
  signal, NOT the validated RLVR claim required by the `rlvr-live-run-not-yet-gated`
  ledger item. `candidateOnly=true`, `level3Evidence=false`, `canClaimAGI=false`.

## Defect found during verification (fixed) — stale-eval ingestion
The auto-run's SSIL ingest + objective gate selected the adapter-eval with
`find … | head -1`, whose ordering is non-deterministic. On the runner it returned a
**stale committed eval** (`mr9sr03clgpk5g.rlvr.adapter-eval.json`, repo-head `bee35a7`,
from an earlier run) **before** this run's fresh `5d62…` eval — the exact trap the
`rlvr-runpod.yml` comment warns about. Effect: the SSIL report the workflow committed
(`8e52541`) carried the OLD run's number (`after=0.7819`) instead of this run's
(`after=0.7404`).
- **Correction:** re-ingested THIS run's fresh `5d62…` eval locally (deterministic
  post-processing of the real pod artifact) and committed the corrected
  `ssil-layer1-real.public-report.json` (`after` 0.7819→0.7404).
- **Root-cause fix:** `runpod-grpo-30m.spec.json` now selects the eval by newest mtime
  (`ls -t … | head -1`) in both the ingest step and the objective gate, so the freshly
  copied-back eval is always chosen. (The copy on the main branch should pick up the same
  fix on its next sync.)
- **Tier impact: none.** The blocker is duration-gated; the 2489s of real GRPO is
  independent of which eval the gate ingested. The objective gate (a parseable fresh
  adapter-eval exists) is also satisfied by the real `5d62…` eval.

## Why this is not gaming
The 23-identical-pass padding approach was rejected in `HONEST-WALL-2026-06-27.md` §3.
This run is the alternative that doc routed to: a single, genuinely long, real GPU
training run whose wall-clock is set by actual computation (dep build + 9B download +
113 GRPO steps + held-out eval), measured by the harness on the box that has the compute.
