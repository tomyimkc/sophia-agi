# Mac Studio operator report — A3 philosophy council teacher + Mac-MLX bench

**Date:** 2026-07-02 · **Box:** Mac Studio (M3 Ultra, Apple Silicon, MLX) · **From:** fresh `origin/main` (c31cc162)
**Guardrails:** `canClaimAGI=false`, `candidateOnly=true` throughout; nothing promoted; `lint_claims` clean; never committed to `main`; Spark GPU and the w1–w5 branch untouched.

## 1. Self-hosted runner (tasks 1–2) — NOT registered (owner-gated)
No GitHub Actions runner is registered on this Mac (`total_count: 0`); the two `mac-mlx-bench` runs
(28562740964, 28562740310) remain **queued** on main. The runner package (v2.335.1, osx-arm64) was
downloaded + extracted, but **registering an unattended self-hosted runner was denied by the
auto-mode safety classifier** as a high-severity persistent autonomous-execution surface that a
relayed "approve all" does not authorize. I did **not** work around it. **Registration is left to the
owner** (`./config.sh … && nohup ./run.sh &`), or a `Bash(./config.sh:*)`+`Bash(./run.sh:*)` permission rule.

**Alternative used (delivers the deliverable without the persistent runner):** the `mac-mlx-bench.yml`
test logic was run **directly on this Mac** (transient local compute) — featurizer suite + readiness
report + R4 ablation. Results below.

## 2. Featurizer readiness (task 2) — READY
`readiness.json`: **`hiddenStateFeaturizerReady: true`** on this box ✅ · `energyHeadFeaturizerReady: false`
(energy-head seam still a stub — expected). Featurizer suite `tests/test_hidden_state_featurizer.py`:
**4 passed, 1 skipped, 1 FAILED**.

**The one failure (captured, NOT patched — per instruction):**
`test_real_hidden_state_featurizer_shape_and_determinism` (line 90):
```
assert abs(norm - 1.0) < 1e-3, "features must be L2-normalized"
AssertionError: 0.0010202049310603645 < 0.001   (norm = 1.0010202049310604)
```
The featurizer **is** L2-normalizing; the output norm is **1.00102**, exceeding the strict `1e-3`
tolerance by ~2e-5 — almost certainly float32/MLX accumulation precision, not a broken featurizer.
`agent/activation_probes.py` was **not** modified. Recommend the cloud decide: loosen the test
tolerance to ~2e-3, or tighten the normalization epsilon.

## 3. A3 philosophy council teacher (task 3) — CANDIDATE, regresses → not promotable
Two-stage MLX SFT (`tools/train_council_teacher.py --seat philosophy`), plan validated first, then run:
stage1 reasoning-SFT (300 iters) → stage2 tool-continued (500 iters), Qwen2.5-3B-Instruct, LoRA
(6.65M params). stage2 **val loss 0.630**, train loss →0.0 (memorization on the small SFT set).
Adapter (CANDIDATE): `training/mlx_adapters/sophia-philosophy-3b`.

**Eval ladder (`--backend mlx --domains philosophy`, 9 cases), CONTENT channel:**

| rung | score | pct |
|---|---|---|
| base | 7/9 | 77.8% |
| base+gate | 7/9 | 77.8% |
| **adapter** | **5/9** | **55.6%** |
| adapter+gate | 5/9 | 55.6% |

**Δ(adapter − base) = −2/9 = −22.2 pts → REGRESSION.** Fails the promotion rule (no useful-correctness
regression). `promote_adapter.py` would reject; **not promoted**. Likely small-data overfitting
(train loss →0). Artifacts: `eval-ladder-baseline.json`, `eval-ladder-adapter.json`.

## 4. R4 claim-router ablation — null marginal contribution
`tools/run_ablation_sophia.py` on the 18-case abstain pack (`abstain-pack-unambiguous-split-2026-06-27`),
local MLX Qwen2.5-3B, modes `sophia-full` vs `sophia-claim-router`:
- sophia-full: score 5/36 (13.89%), 0 passed.
- **claim-router delta vs full: 0.0** (scorePctDelta 0.0, `meaningfulMargin: false`).
- `falsificationCheck.evaluable: false` (needs a raw-model arm in the same run).

The claim router adds **no measurable marginal contribution** on this pack. Evidence only — no
promotion of `use_claim_router` to default-on (that must clear `evaluate_update()` with
answerable-coverage as a protected metric). Artifact: `claim-router-ablation.public-report.json`.
