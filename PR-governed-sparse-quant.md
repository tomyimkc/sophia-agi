# PR — feat(sparse-quant): governed adaptive quant, IndexShare, expert-offload, QAT study

**Branch:** `feat/governed-sparse-quant` → `main`
**Commit:** `c12721e` (pushed)
**Create the PR here:** https://github.com/tomyimkc/sophia-agi/compare/main...feat/governed-sparse-quant

Paste the body below into the PR description.

---

## What

Seven governed scaling artifacts that make GLM-5.2-class aggressive sparsity/quantization adoptable **honestly** — each built to the repo's `offline_invariants() -> (ok, dict)` discipline, with an explicit `honest_scope` caveat and deferral to the no-overclaim gate. They are numpy/pure-Python references (CI-tested for policy correctness), not GPU deployments.

This answers the goal *"build a highly-intelligent model: large params, trainable with few compute resources, low RAM at release"* within the charter boundary — via **sparsity + governed quantization**, not dense scale or frontier pretraining.

## Why

The GLM-5.2 analysis showed the techniques that make large models cheap-to-serve (MoE sparsity, adaptive per-tensor quant, IndexShare, expert offloading) are largely *architectural/quant choices*. Sophia's existing trust-governor pattern (equivalence proofs, `offline_invariants`, the no-overclaim gate, decontamination) is exactly what makes them safe to adopt. These artifacts make the efficiency primitive **carry its own proof** — the `Governed-Scaling.md` thesis in concrete code.

## The artifacts

| File | What | Invariants |
|---|---|---|
| `moe/adapt.py` | Adaptive mixed-precision quant — sensitivity + greedy budgeted bit-allocator with protected floor (reproducible Unsloth Dynamic 2.0 principle) | 12 ✓ |
| `moe/calibrate.py` | Calibration-distribution matching — deployment-sourced, eval-disjoint (via existing contamination guard), datasheet | 5 ✓ |
| `kernels/indexshare.py` | IndexShare reproduction — cross-layer index amortization with measured quality-vs-compute curve | 9 ✓ |
| `serving/expert_offload.py` | Tiered expert offloading — promote-on-route governance (only active experts resident) | 7 ✓ |
| `serving/kv_quant.py` | INT8/INT4 KV-cache quant — content-deterministic scale so prefix sharing survives | 7 ✓ |
| `pretraining/qat/study.py` | QAT-on-known-floor study — does ternary-pushing training lower the quant gap vs closed-form floor? | 6 ✓ |
| `pretraining/architecture/run_sparse_quant.py` | Capstone: sparsity + adaptive quant composed end-to-end against the known floor | runs ✓ |

## The boundary doc

`docs/11-Platform/Cheap-Compute-Boundary.md` states the honest scope and prevents the overclaim the feasibility doc was written to kill. Three boundaries: (1) large *total* via sparsity, never dense scale; (2) cheap **adaptation**, never frontier pretraining; (3) low-RAM **with measured error bounds**. Includes a one-line test every future claim must pass.

## Does NOT duplicate open PRs

Checked PRs #129 and #131 first (per the handover doc). **PR #131 already adds NVFP4 to `moe/quant.py`** — but it is *uniform* quantization. This PR's `moe/adapt.py` is the complementary *adaptive allocation* layer, so they **compose** rather than conflict. Neither PR touches adaptive allocation, calibration-matching, IndexShare, expert-offloading, KV-quant, or QAT.

## Verification

- All 7 module `offline_invariants()` **PASS**.
- **64/64 tests pass** across 5 new + 3 adjacent existing test files.
- Package imports (`moe`, `serving`, `kernels`) resolve with new exports.
- No regressions in adjacent tests (`test_moe.py`, `test_flash_attention.py`, `test_kv_serving.py`).
- Three bugs caught and fixed by the invariants process before completion (floor-infeasibility fail-closed; expert-offload double-delete; QAT per-layer scale).

## Honest scope

Every artifact defers to the no-overclaim gate for any capability claim. These are **mechanisms with bounded-error proofs**, not validated capability results. The next step (P6, not in this PR) is a real-GPU validation run to the κ ≥ 0.40 / 2-judge bar.

`canClaimAGI` stays **false**.
