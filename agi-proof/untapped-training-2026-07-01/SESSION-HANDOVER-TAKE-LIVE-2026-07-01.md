# Session handover — W1–W5 take-live (2026-07-01)

> **candidateOnly:true · canClaimAGI:false.** All five failure-ledger rows **remain Open**;
> no gate was cleanly cleared. Two live gates landed as **strong candidates**, honestly not
> flipped. Branch `feat/agi-proof-candidate-tools`, 4 commits pushed (`b07e2b9b..904954d6`).

## What happened

Took the W1–W5 untapped-training **instruments** (from `b07e2b9b`) as live as an M4 Max
(local MLX; grok-CLI cloud only) honestly allows.

1. **All 5 offline** — ran + adversarially verified (31 tests green; no mock, no overclaim).
   Report: `EXECUTION-REPORT-2026-07-01.md`. Registered the 5 Open rows in
   `agi-proof/failure-ledger.md`. Fixed a cosmetic DRY/LIVE note bug in
   `tools/adversarial_gate_selfplay.py`. (commit `1c3fb8b3`)
2. **W2 calibration — LIVE** (commit `d5f2eeda`): calibration-SFT via `mlx_lm lora` on
   Qwen2.5-3B. Held-out (sympy-checkable math, decontam clean) **ECE 0.200 → 0.058/0.087**
   across 2 seeds at matched-or-better accuracy; **acc@coverage 0.75 → 1.00**. Seed-1 ΔECE
   95% CI includes 0 (N=80) + narrow surface → **row Open**.
   Artifact: `agi-proof/benchmark-results/w2-calibration-sft/`.
3. **W1 verifier-PRM — LIVE** (commit `148bebed`): implemented the gate's named seam
   `agent.activation_probes.build_hidden_state_featurizer(spec="mlx", model, tok)` (real
   2048-d residual stream; fail-closed default preserved → 36 tests green). Within-domain
   held-out agreement **0.73 math / 0.90 physics** (vs 0.41/0.50 degenerate) — but
   **held-out-DOMAIN (math→physics) = 0.50 chance** + RLVR half unrun → **row Open**.
   Artifact: `agi-proof/benchmark-results/w1-verifier-distilled-prm/`.
4. **W5 probe-as-loss — methodology LIVE, coupling NOT rushed** (commit `904954d6`): reused
   the featurizer; base model separates honest/deceptive DPO text **perfectly on real hidden
   states (1.0/1.0, gap 0.0)**. The probe-as-loss LM coupling is left as a careful next step
   (rushing the highest-risk Goodhart tool is its own failure mode) → **row Open**.
   Artifact: `agi-proof/benchmark-results/w5-probe-as-loss/`.

## Key binding facts discovered

- **mlx_lm 0.31.3 ships NO dpo/orpo trainer** (only `lora`/`dora`/`full`). W2's "existing DPO
  path" wiring is unavailable locally; faithful wiring is calibration-SFT.
- **`build_hidden_state_featurizer` is now real** for `spec="mlx"` when passed a loaded
  `(model, tokenizer)`; it still raises `RuntimeError` with no model (fail-closed, keeps
  `test_truth_probe` + `test_probe_representation_training` green).

## Reproduce (this box)

- Working venv (has sympy/numpy/pytest/mlx/mlx_lm): a `python3.12 -m venv` under the session
  scratch — `python3.12` itself is uv-managed (pip blocked). macOS has no `timeout` (use a
  Python thread timeout). Qwen2.5-3B-Instruct is HF-cached (5.8GB); LoRA peak ~8.5GB.
- Scratch harnesses (not committed): `w2_calib_pilot.py`, `w1_prm_live.py`, `w5_probe_live.py`.

## Exact next steps (none done)

- **W5 probe-as-loss coupling** (highest value, now unblocked): custom MLX LoRA loop, aux loss
  `BCE(loss_probe(mean_pooled_hidden(text)), honesty_target)` on a headroom surface (truncated
  claims, base sep ~0.80); after training re-featurize test with the adapted model, require the
  **disjoint audit probe** to improve + `goodhartGap ≤ 0.15` + a from-scratch post-run audit to
  still separate. A Goodhart negative is a valid result.
- **W3 provenance-weighted**: weighted-vs-uniform SFT + a leave-one-out slice + register-collapse
  check (fully local).
- **W4 adversarial self-play**: ≥3 grok rounds with between-round LoRA retrain on mined
  fabricate-and-pass negatives (local, ~20s/grok call).
- **W2/W1 to close their gates**: larger held-out N + a ≥3rd seed + a broader non-math surface
  (W2); mixed-domain PRM tested on a held-out THIRD domain + PRM-as-RLVR-reward on GPU (W1).
