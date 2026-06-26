# Handover — governed sparse-quant work, to the next AI

**From:** GLM-5.2 session 2 (2026-06-26). **Branch:** `feat/governed-sparse-quant` (pushed, commit `c12721e`). **PR not yet opened** — `gh` is not installed on this box; see `PR-governed-sparse-quant.md` for the ready-to-paste PR body and the compare URL.

**Read this whole document before acting.** This is still a **multi-agent repo**: PRs #129 and #131 are open and overlap the sparse-quant territory. Coordination — not just continuation — is the job.

---

## ⚠️ FIRST: PR OPENING (the one human action this session couldn't do)

The branch is pushed. Open the PR by visiting:
```
https://github.com/tomyimkc/sophia-agi/compare/main...feat/governed-sparse-quant
```
Paste the body from `PR-governed-sparse-quant.md`. **Do not re-run or re-commit** — the work is done and tested (64/64 pass). Your first job is opening the PR, then reading the merge-conflict map below.

---

## WHAT SHIPPED (committed `c12721e`; do NOT redo)

Seven governed scaling artifacts + a boundary doc, all built to the repo's `offline_invariants() -> (ok, dict)` discipline. Every module PASSes its invariants; 64/64 tests pass.

| File | What | ✓ checks |
|---|---|---|
| `docs/11-Platform/Cheap-Compute-Boundary.md` | **The boundary doc** — honest scope of "few compute resources": large-total-via-sparsity (not dense), cheap **adaptation** (never frontier pretraining), low-RAM **with measured error bounds**. Has a one-line test every future claim must pass. | — |
| `moe/adapt.py` | Adaptive mixed-precision quant — sensitivity measurement + greedy budgeted bit-allocator with a **protected floor** for critical tensors. The reproducible Unsloth-Dynamic-2.0 principle. | 12 |
| `moe/calibrate.py` | Calibration-distribution matching — source calibration from deployment data, **prove eval-disjoint via the existing `provenance_bench.dataset_guard`**, emit a datasheet. | 5 |
| `kernels/indexshare.py` | **IndexShare reproduction** — cross-layer sparse-attention index amortization (GLM-5.2's attention innovation) with a measured quality-vs-compute curve. | 9 |
| `serving/expert_offload.py` | Tiered MoE expert offloading with **promote-on-route governance** — only the active expert set resident in fast memory. | 7 |
| `serving/kv_quant.py` | INT8/INT4 KV-cache quant with a **content-deterministic per-block scale** so prefix sharing survives quantization. | 7 |
| `pretraining/qat/study.py` | **QAT-on-known-floor study** — does ternary-pushing training lower the quant gap against the closed-form ground-truth floor `E = source_entropy(src)`? | 6 |
| `pretraining/architecture/run_sparse_quant.py` | Capstone: sparsity + adaptive quant composed end-to-end against the known floor. Runs clean (E=0.522, quant gap=0.128). | runs |

Exports wired into `moe/__init__.py`, `serving/__init__.py`, `kernels/__init__.py`. Tests: `tests/test_adapt.py`, `test_calibrate.py`, `test_indexshare.py`, `test_qat.py`, `test_serving_quant.py`.

---

## ⚠️ MERGE-CONFLICT MAP (resolve before merging anything)

The prior handover (`HANDOVER-FROM-GLM5.2.md`) flagged PRs #129 and #131 as overlapping. **My work was deliberately scoped to NOT touch their files** — I read both PRs first and built only complementary surfaces. But on merge you still need to check:

| File | This branch | PR #131 (NVFP4 Spark) | Conflict? |
|---|---|---|---|
| `moe/quant.py` | **not touched** | adds NVFP4 (E2M1 + FP8 micro-scale) | **NONE — complementary.** #131 = uniform quant; my `moe/adapt.py` = adaptive allocation. They compose. |
| `moe/__init__.py` | modified (added adapt/calibrate exports) | not touched | NONE expected |
| `kernels/__init__.py` | modified (added indexshare exports) | not touched | NONE expected |
| `serving/__init__.py` | modified (added expert_offload/kv_quant exports) | not touched | NONE expected |
| `.github/workflows/spark-gpu.yml` | **not touched** | modified | N/A (not my file) |
| `docs/11-Platform/Spark-Local-GPU-Lane.md` | **not touched** | modified | N/A |

**Action:** my branch should merge cleanly to `main` and to #131 (different files). The risk is `moe/__init__.py` if #131 also edits it — check on merge. **Read PR #131's NVFP4 before any work that touches `moe/quant.py`** — it's the uniform-quant half of what my adaptive layer allocates across.

---

## THE NON-NEGOTIABLE DISCIPLINE (do not weaken)

No-overclaim standard (`RESULTS.md` + `tools/lint_claims.py`, CI-enforced). A number is VALIDATED only with: ≥2 independent judge families (judge ≠ gate ≠ subject), κ ≥ 0.40, ≥3 runs, 95% CIs excluding zero. Everything else = "illustrative"/"candidate-only". `canClaimAGI` stays **False**.

**These seven artifacts are MECHANISMS WITH BOUNDED-ERROR PROOFS, not validated capability results.** Every one carries an `honest_scope` caveat and defers to the no-overclaim gate. Do not let anyone quote them as "Sophia can now run at X% with Y RAM" — that requires the P6 real-GPU validation (below), which has NOT been done.

**The boundary doc (`Cheap-Compute-Boundary.md`) is the standing instruction.** Its one-line test: *does a claim survive substituting "adapted" for "trained" and "within a measured error bound on a decontaminated eval" for "retained capability"?* If no — it does not ship.

---

## WHAT'S GENUINELY UNRESOLVED (ranked by EV)

### Tier 1 — the one missing validation (highest EV)
**P6: real-GPU validation of the quantized path to the no-overclaim bar.** Today every artifact is a numpy/pure-Python reference with bounded-error invariants — none has been run on a real model against FP16 on a held-out set. The first honest "low-RAM, capability-retained" claim needs:
- A real quantized model (start with `moe/adapt.py` allocation applied to a small real base, e.g. Qwen2.5-3B, via the existing QLoRA path in `tools/train_lora.py`),
- Evaluated against FP16 on a held-out, **decontaminated** set (use `moe/calibrate.py`'s disjoint check + the existing eval ladder),
- To the κ ≥ 0.40 / 2-judge-family / ≥3-seed bar.
- **Target hardware:** the DGX Spark the human is deploying (see `docs/11-Platform/Spark-Local-GPU-Lane.md`; PR #131's NVFP4 is the Spark-native quant path — compose with it).

### Tier 2 — composition studies (cheap, on-substrate)
- **Run the IndexShare quality/compute curve at the block sizes that matter.** `kernels/indexshare.py` sweeps `group ∈ {1..8}` on a 6-layer toy. Extend to measure where the error budget is exceeded for a *real* layer-count, and whether adaptive re-indexing (re-index when cross-layer index divergence > ε) recovers the loss — the "adaptive sharing" idea from the analysis, currently unimplemented.
- **Connect `moe/router.py` to a trainable nano MoE end-to-end** (the P7 "bridge"). The capstone `run_sparse_quant.py` uses the existing `pretraining/architecture/moe.py` toy, but `moe/router.py` (the numpy reference with capacity + aux loss) is still disconnected from any trainable model. Bridge them: a nano MoE whose routing uses `MoERouter`, then measure load-balance + sparse-vs-dense at matched active compute.

### Tier 3 — wiring into the docs
- Add the new modules to `docs/11-Platform/Systems-Track.md`'s module table and test table (the JD-mapping doc). FlashAttention was the first paper reproduction; IndexShare is the second — document it there.
- Reference `Cheap-Compute-Boundary.md` from `VISION.md`'s "Governed scaling" bullet so the boundary is discoverable.

---

## BUGS ALREADY CAUGHT AND FIXED (don't re-litigate)
The `offline_invariants()` process caught three before completion:
1. **`moe/adapt.py`** — protected-floor seeding can overshoot a low target budget → made it fail-closed with an explicit `ValueError` instead of silently overspending.
2. **`serving/expert_offload.py`** — `_enforce_gpu_budget` had a double-delete (`popitem` then `_move` both removed from the dict) → `_move` now owns all dict removal.
3. **`pretraining/qat/study.py`** — ternary regularizer needed per-layer scales (single scalar can't zero both weight matrices) → accepts a `dict` now, matching real BitNet.

---

## INFRA NOTES (don't repeat prior mistakes)
- **`gh` is NOT installed on this dev box.** PRs must be opened via the web compare URL, or install `gh` first. The PR body is ready in `PR-governed-sparse-quant.md`.
- **This dev box CANNOT SSH to RunPod pods** (network blocks non-443 TCP). RunPod work goes through the **GitHub Action** (`rlvr-runpod.yml`) using the repo's `RUNPOD_API_KEY` secret. The DGX Spark, once deployed, is local.
- **Two pre-existing test failures are NOT mine:** `tests/test_code_verifier.py` and `tests/test_generate_math_code_curriculum.py` both fail on `import resource` (Unix-only module) on this Windows box. Ignore them; they're environment, not code.
- **Spot pods get preempted mid-pip-install.** Use on-demand for any real GPU run; wide GPU list dodges capacity-500.

---

## ONE-LINE BOTTOM LINE FOR THE HUMAN

Seven governed sparse-quant artifacts + a boundary doc are committed (`c12721e`), pushed, and tested (64/64). **The PR is not opened — `gh` isn't installed; open it via the compare URL with the body in `PR-governed-sparse-quant.md`.** The next AI's first job is **open the PR**, then the **P6 real-GPU validation** (the one missing piece that turns these mechanisms into the first honest "low-RAM, capability-retained" claim) — composed with PR #131's NVFP4 on the DGX Spark.
