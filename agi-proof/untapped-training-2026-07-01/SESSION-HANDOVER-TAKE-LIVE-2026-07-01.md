# Session handover — W1–W5 take-live (2026-07-01)

> **candidateOnly:true · canClaimAGI:false.** All five failure-ledger rows **remain Open**;
> no gate was cleanly cleared. **All five instruments were taken live locally** (W1–W5); the
> live gates landed as **candidates / one clean Goodhart negative**, honestly not flipped.
> Branch `feat/agi-proof-candidate-tools` (`b07e2b9b..601d3d15`).

> **Round 4 (2026-07-02): promotion experiments — strongest honest evidence, all rows still
> OPEN.**
> - **W2 second surface** (non-math letter-counting): ECE 0.53→0.43 + acc@cov 0.4→1.0
>   directionally like math, BUT ΔECE CIs include 0 (N=30) + a capability confound (accuracy
>   jumped 0.47→0.83). Directional support, not a clean promotion. Math surface stays strongest.
> - **W4 promotion** (2 seeds, STRICT fabricate-and-pass, harder guardrail): strict fab-and-pass
>   0.94/0.81→0.0 both seeds; hard-answerable acc INCREASES (no over-abstention); novelty>floor.
>   Gate conditions met with the strict metric + 2 seeds.
> - **W5 rigorous** (N=120, per-seed paired CIs): 2/3 seeds' disjoint-audit-delta CI EXCLUDES 0
>   (seed0 0.617→0.90; seed1 0.90→0.95); seed2 grazes 0. Genuine anti-Goodhart method, near-close.
> - **W1** (infra-bound) and **W3** (in-sample limit) unchanged — at their honest local ceilings.

> **Round 3 (2026-07-02): gate-closing next steps implemented for all five. All five rows
> REMAIN OPEN** — W2 and W4 met their gate conditions on controlled surfaces but are recorded as
> **candidates (not promoted)** per the no-overclaim contract (maintainer sign-off required for
> any claim-status flip; the user chose to keep them Open).
> - **W2 — candidate, Open** (commit `e21831dc`): larger run (decontaminated held-out N=195, **3
>   seeds**) — every ΔECE 95% CI excludes 0, accuracy-at-coverage 0.79→0.99 (controlled math
>   surface). Gate conditions met; promotion pending a broader non-math surface + sign-off.
> - **W4 — candidate, Open** (commit `520809a9`): v2 MIXED objective (abstain-on-unanswerable +
>   answer-correctly-on-answerable) + adaptive proposer FIXES the over-abstention — fabrication
>   0.375→0.0, answerable acc stays 1.0, novelty 0.75–1.0. Gate conditions met; promotion pending
>   seeds/harder guardrail + sign-off.
> - **W3 strengthened, still Open** (commit `520809a9`): true per-example weighted-loss (custom
>   loop, not replication) → weighted 1.0 vs uniform 0.45–0.55, 2 seeds. Eval is in-sample (no
>   held-out generalization) → Open.
> - **W1 Open, infrastructure-bound** (commit `60dd1521`): 3-seed characterization — held-out-
>   DOMAIN agreement ~chance both directions (0.488 / 0.495). Only 2 verifier domains exist +
>   no GPU RLVR → cannot close locally (needs a 3rd domain or GPU). Not a method failure.
> - **W5 Open, strong anti-Goodhart signal** (commit `2336d22b`): ENSEMBLE probe-as-loss (K=4
>   independent probes) REVERSES v1's gaming — disjoint audit improves on all 3 seeds (mean
>   +0.106). But seed0 Δ is noise + no per-seed CI → not a rigorous close.

> **Round 2 (same session): W3, W4, and the W5 probe-as-loss coupling also taken live.**
> - **W3** (commit `3c5e82cb`): conflicting-provenance experiment — provenance-weighted **0.95**
>   vs uniform **0.50** correct; influence proxy agrees with LOO (de-poisoned slice 0.25→0.75);
>   no register collapse. Controlled synthetic surface → **row Open**.
> - **W4** (commit `fbebfa33`): multi-round self-play — held-out fabricate-and-pass **0.375→0.0**,
>   BUT over-abstention side effect (2/5 answerable Qs refused) + fixed (non-adaptive) proposer →
>   **row Open**.
> - **W5 coupling** (commit `601d3d15`): custom MLX gradient-through-probe LoRA loop → **GOODHART
>   DETECTED** (loss-probe 0.867→0.956, disjoint audit 0.756→0.711, gap 0.244>0.15; the audit
>   correctly refused to certify). Valid negative → **row Open**. Reusable footgun found: `mlx_lm
>   lora` fails when the valid split has fewer rows than `--batch-size` (use batch-size ≤ valid).

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

## Exact next steps (to CLOSE the still-Open gates)

All five instruments are now taken live; each row stays Open on a specific, documented gap:
- **W2**: larger held-out N + a ≥3rd seed + a broader non-math verifier-checkable surface.
- **W1**: mixed-domain PRM tested on a held-out THIRD domain; PRM-as-dense-RLVR-reward on GPU.
- **W3**: real held-out generalization suite + ≥2 seeds + true per-example loss weighting (not
  replication) + a real TracIn/influence backend validated vs full LOO.
- **W4**: an ADAPTIVE proposer (novelty-per-round vs a floor) + a DPO objective (abstain-on-trap
  preferred over fabricate, correct preferred over abstain) so fabrication drops WITHOUT
  over-abstention + an answerable-accuracy guardrail.
- **W5**: a probe-as-loss variant that improves the AUDIT probe too (multi-probe/ensemble loss,
  adversarial-probe co-training, or steering to a held-out honesty target) with goodhartGap≤0.15.
  The coupling built here validated the *audit* (it triggered it); closing needs a coupling that
  *passes* it. Each artifact's `.candidate.json` has the exact `nextToClose`.
