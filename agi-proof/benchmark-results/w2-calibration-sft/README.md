# W2 calibration-SFT — local-MLX take-live (2026-07-01 pilot → 2026-07-02 gate MET-scoped)

> **UPDATE 2026-07-02 — gate CONDITIONS met on the math surface; recorded as CANDIDATE, row
> stays OPEN** (per the no-overclaim contract — not promoted without maintainer sign-off + a
> broader surface). On a **decontaminated** held-out (N=195, sympy-checkable), **all 3 seeds**
> lower ECE (base 0.231 → 0.060/0.083/0.058) with every per-seed ΔECE 95% CI **excluding 0**, at
> matched-or-better accuracy and **accuracy-at-coverage 0.79 → 0.98/0.99/0.99** (no hedging
> collapse), one pinned commit. **SCOPE: controlled verifier-checkable MATH surface, type-level
> confidence via calibration-SFT — NOT general calibration, NOT level3Evidence, NOT canClaimAGI.**
> Artifact + sha256: `w2-calibration-sft-CLOSE-2026-07-02.json` (`promotedToLedger:false`).

> **candidateOnly:true · level3Evidence:false · canClaimAGI:false** (original pilot, 2026-07-01;
> N=80, seed-1 CI included 0). Machine-readable result + sha256:
> `w2-calibration-sft-pilot-2026-07-01.candidate.json`.

## What was done

Took the W2 proper-scoring calibration objective **live** through the repo's real local
training surface (`mlx_lm lora` SFT on `Qwen/Qwen2.5-3B-Instruct`, the model in the v3 run
record). **Binding fact:** mlx_lm 0.31.3 ships **no DPO/ORPO trainer** (only `lora`/`dora`/
`full`), so W2's "existing DPO path" wiring is unavailable locally; the faithful wiring is
**calibration-SFT** — teach the model to state confidence == its own verifier-measured
per-difficulty accuracy.

- **Surface:** freshly-generated, **sympy-checkable** arithmetic/linear math (objective
  correctness, unlimited disjoint N). Chosen over provenance facts because
  `data/attributions.json` is ledger-flagged decontamination-exhausted.
- **Splits:** train seeds 1 & 3 (N=120 each), held-out seed 2 (N=80). Train↔held-out
  exact-overlap = **0**.
- **Objective:** target `Confidence` = Laplace-smoothed per-kind train accuracy
  (add/mul ≈ 0.95, `two_step` ≈ 0.30, linear ≈ 0.72). Measured by the repo's own
  `agent.calibration.expected_calibration_error`.
- **Two independently-trained seeds** (train-data seed + mlx seed both varied), 300 LoRA
  iters each, pinned to one commit.

## Result (held-out N=80)

| | ECE | accuracy | mean conf | acc@coverage 0.5 |
|---|---|---|---|---|
| **base** | 0.200 | 0.800 | 1.00 (constant) | 0.75 |
| **seed 0** | **0.058** | 0.787 | 0.803 | **1.00** |
| **seed 1** | **0.087** | 0.850 | 0.809 | **1.00** |

- Both seeds **lower held-out ECE** at **matched-or-better accuracy**, and
  **accuracy@coverage rises 0.75 → 1.00** — no verbal-hedging collapse; the confidence is
  now *discriminative* (the base model had a single constant confidence of 1.0).
- Paired bootstrap ΔECE: seed 0 **+0.142, 95% CI [+0.036, +0.202] excludes 0**; seed 1
  **+0.113, 95% CI [−0.005, +0.214] includes 0** (N=80 underpowered for the smaller effect).

## Why the gate is NOT cleared (row stays Open)

1. **seed 1's per-seed ECE-reduction 95% CI includes 0** — the ≥2-seed reproduction is
   directionally consistent but not both-seeds-significant at N=80.
2. **Narrow surface.** Confidence is type-level (6 values) over a controlled math generator,
   not general open-domain calibration. It demonstrates the *measurement→learning wiring*
   works end-to-end locally and lowers held-out ECE; it is not yet general calibration.

## To close it

Larger held-out N (tighten each seed's CI to exclude 0), a ≥3rd seed, and a broader
non-math verifier-checkable surface (open-domain factual QA with a real answer-verifier and
a per-instance — not per-type — confidence signal).

## Second surface (2026-07-02) — letter-counting: directional support, not a clean promotion

To rule out a math-taxonomy-specific effect, the calibration-SFT was re-run on a **non-math**
verifier-checkable surface (letter-counting, per-instance confidence target). Result
(`w2-second-surface-letters-2026-07-02.candidate.json`): ECE drops (base 0.533 → 0.43/0.458/0.458)
and accuracy-at-coverage rises (0.40 → 1.0 all 3 seeds) — **directionally the same as math**. But
this is **not a clean promotion**: the per-seed ΔECE 95% CIs **include 0** (N=30 held-out
underpowered), and there is a **capability confound** (the SFT taught the correct answers, so
accuracy jumped 0.467 → 0.83, mixing calibration with a capability gain — unlike the math surface
where base accuracy was already high). The strongest W2 evidence remains the **math** surface
(3 seeds, all ΔECE CIs exclude 0). Row stays **Open**.
