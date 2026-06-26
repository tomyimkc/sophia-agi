# Pre-registration — Live RL Milestone 1 (first gated GRPO weight update)

**Status:** PRE-REGISTERED (2026-06-26) · **OPEN ledger item:** `rlvr-live-run-not-yet-gated-2026-06-21`
**Plan:** `docs/06-Roadmap/frontier-readiness/05-live-rl.md` · **Wave 1 #1.**

This file fixes the hypothesis, method, and pass/fail gate **before** the GPU run, so the result
cannot be retrofitted. A within-noise or null outcome is a valid, logged result — not a reason to
weaken the gate.

## Hypothesis
Online GRPO with the repo's **verifiable provenance gate as the reward** (RLVR, abstention-positive)
produces a **positive held-out before/after delta** in provenance faithfulness on an entity-disjoint
split, **without** regressing true-author cases (no FP-regression) and **without** abstention collapse.

## Fixed method (pre-registered)
- **Base model:** `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) — standardized; LoRA on
  `q,k,v,o_proj` + `gate_proj,up_proj,down_proj`.
- **Algorithm:** GRPO (TRL `GRPOTrainer`), `--task provenance --reward gate`, `--num-generations 8`,
  `--beta 0.04`, `--lr 1e-5`, `--epochs 3`, **seeds {0,1,2}**.
- **Reward:** judge-free gate reward for the *training* signal (LLM judges never grade training
  targets); LLM judges used **only** on the held-out *eval* semantic axis, ≥2 distinct vendor families
  (both ≠ Qwen).
- **Eval:** `tools/eval_rlvr_adapter.py --mode real --task provenance` on the entity-/family-disjoint
  holdout; report `meanReward`, `pass@1`, and `trueFalsePositiveRate` delta.
- **Compute:** 1×A100/H100 80GB (bf16, vLLM colocate) on DGX Spark **or** RunPod; auto-teardown via
  `tools/runpod_rlvr.py` (`finally`-delete + watchdog). Budget ≈ 6 GPU-hr ≈ $15–30 if rented.

## Pass/fail gate (decided now)
**PASS (close the OPEN item)** iff **all** hold:
1. mean Δ(meanReward) **> 0** across 3 seeds, **95% bootstrap CI excludes 0**;
2. `provenance_bench.aggregate._is_validated` true: `notMock ∧ ≥2 judge families ∧ κ≥0.40 ∧ ≥3 runs`;
3. **no FP-regression**: `trueFalsePositiveRate` delta ≤ 0 (`false_positive_regressions == []`);
4. **no abstention collapse**: held-out abstention rate does not fall vs base;
5. contamination-free: `entity_intersection == [] ∧ family_intersection == []` asserted in-report.
Else → record the honest within-noise/null in the ledger; do **not** weaken any gate.

## Integrity curves to log
reward vs step (monotone→plateau), **KL-to-ref vs step** (bounded), completion-length vs step
(no length-hacking), grad-norm. Register the adapter (`weightsSha256`, `trainingConfigHash`,
`datasetManifest`) in `agi-proof/mlops/checkpoint-registry.json`.

## Launch blockers (provision first — M0 confirmed the seam, these gate M1)
`HF_TOKEN`, `RUNPOD_API_KEY`, **≥2 judge-vendor keys**. With only one vendor key, the run can clear the
**judge-free rung** but **not** the validated `_is_validated` headline — and the report must say so.

## M0 (offline contract) — DONE, green (2026-06-26)
`agent/gate_reward.py` ✓ · `run_rlvr.py --model mock {provenance,math}` ✓ ·
`eval_rlvr_adapter.py --mode mock` provenance ✓ (math mock not-passed — strict-gate behavior, to confirm).
