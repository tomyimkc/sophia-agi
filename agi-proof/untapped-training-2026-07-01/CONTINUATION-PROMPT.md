# Continuation prompt — take W1–W5 (untapped training signals) live

You are a terminal-native coding agent (Claude Code or equivalent) working directly in the
`tomyimkc/sophia-agi` git checkout. A research-advisor session built five fail-closed
INSTRUMENTS that each convert one of Sophia's measurement signals into a training signal, but
none has been run against a live model backend. Your job is to take them live in priority
order, and to flip each of five failure-ledger rows from **Open** to a measured verdict ONLY
when its stated acceptance gate is met. Do not overclaim; an instrument that runs is not a
result.

## Ground truth (verify first, do not trust this prompt blindly)

- Clone last seen on branch `docs/session-handover-2026-07-01` @ `544a26fe`. Run
  `git rev-parse --abbrev-ref HEAD && git status --short` and re-confirm the files below
  exist on YOUR HEAD before starting — the advisor placed them as untracked additions and a
  branch may have moved.
- The eleven files (all untracked `??`, nothing of the user's overwritten):
  - `tools/train_calibration_objective.py`      (W2)
  - `tools/distill_process_reward_model.py`     (W1)
  - `tools/provenance_weighted_training.py`     (W3)
  - `tools/adversarial_gate_selfplay.py`        (W4)
  - `tools/probe_representation_training.py`    (W5)
  - `tests/test_*.py` for each of the five
  - `agi-proof/untapped-training-2026-07-01/{README.md,failure-ledger-additions.md}`
- Baseline check — this MUST pass before you change anything:
  ```
  PYTHONPATH=. python3 -m pytest \
    tests/test_distill_process_reward_model.py tests/test_train_calibration_objective.py \
    tests/test_provenance_weighted_training.py tests/test_adversarial_gate_selfplay.py \
    tests/test_probe_representation_training.py -q
  # expect: 31 passed
  ```
  If it is not 31 passed on your HEAD, STOP and report the diff — do not build on a red base.

## Non-negotiable discipline (the whole point of this repo)

1. **Fail closed.** No backend / degenerate input / repo import failure → an environment
   artifact (`ok:false`) or a non-zero exit, NEVER a fabricated metric. Every output already
   carries `candidateOnly:true, level3Evidence:false, canClaimAGI:false`; keep it.
2. **A run is not a result.** Only flip a ledger row when its acceptance gate (below, and in
   `failure-ledger-additions.md`) is met, with the artifact + sha256 written under
   `agi-proof/benchmark-results/`. Otherwise the row stays **Open** and you report the number
   you got plus why it did not clear the gate.
3. **Real backend, not mock.** `agent.model._auto_provider()` returns `"mock"` when no API
   key is present, and mock `.generate()` returns FABRICATED text with `ok=True`. Any live
   run MUST assert `cfg.kind != "mock"` (or an explicit real provider) before trusting a
   number. A metric from a mock backend is a fabrication — treat it as fail-closed.
4. **sympy required for W1.** `agent.math_verifier` needs sympy; without it the verifier
   abstains (`sympy_unavailable`) and W1 emits an environment artifact. `pip install sympy`
   in the training env first.

---

## Run order and per-tool acceptance gates

### 1. W2 — calibration objective (DO FIRST: lowest cost, tightest fit)

Ledger row: `w2-calibration-objective-not-in-lm-training`.

The tool already proves the loss lowers ECE on a FROZEN model's confidences (offline, via the
repo's own `agent.calibration`). Your job is the maintainer seam: wire the proper-scoring +
asymmetric-abstention loss into the LM's own fine-tune.

- The real training surface is MLX LoRA, invoked as (verbatim from
  `training/local_sophia_v2/training_run_mlx_sophia_v3.json`):
  ```
  python3 -m mlx_lm lora --train --model Qwen/Qwen2.5-3B-Instruct \
    --data training/local_sophia_v2/mlx --adapter-path training/mlx_adapters/sophia-vNEXT ...
  ```
  DPO/SFT data lives in `training/{wiki_provenance_dpo,hard_negatives_dpo,...}.jsonl`.
  Inspect `training/qat.py` and the `training/local_sophia_v*/manifest.json` run records to
  see exactly how a run is configured and logged — mirror that, do not invent a new harness.
- Two viable wirings, pick per what `mlx_lm` exposes on your version:
  (a) **preference-style**: convert the calibration objective into DPO-style preferred/rejected
      pairs (confident-correct preferred over confident-wrong; abstain-on-trap preferred over
      confident-wrong) and train through the existing DPO path — least code, no loss surgery;
  (b) **auxiliary-loss**: add the Brier/log + abstention term as an auxiliary loss in a custom
      training loop — more faithful, more work. Prefer (a) first.
- **Acceptance gate to flip the row:** the fine-tuned adapter lowers ECE on a HELD-OUT,
  verifier-checkable eval set at **matched-or-better accuracy-at-coverage** (no verbal-hedging
  collapse — measure accuracy at fixed coverage, not raw abstention rate), reproduced across
  **≥2 seeds** on one pinned commit. Write `calib_before_after.json` + sha256 under
  `agi-proof/benchmark-results/`. If ECE drops but accuracy-at-coverage regresses, the row
  stays Open and you report the hedging tradeoff.

### 2. W1 — verifier-distilled PRM (flagship AGI-capability bet)

Ledger row: `w1-verifier-distilled-prm-not-trained-live`.

The tool distills real per-step labels from `verify_derivation` and measures held-out +
held-out-DOMAIN agreement using a stand-in probe over transparent features. Two seams:

- **Real features:** implement `agent.activation_probes.build_hidden_state_featurizer(spec="mlx")`
  to return true residual-stream vectors from the MLX model, and swap it in for
  `featurize_text` in the PRM path. (This is shared with W5 — do it once.)
- **PRM as reward:** wire the trained PRM as the dense per-step reward in `tools/run_rlvr.py`
  (verify that file's reward hook signature first; the advisor did not modify it).
- **Acceptance gate:** PRM on real hidden states reaches **≥0.80 held-out-DOMAIN agreement**
  with the symbolic oracle, AND as dense RLVR reward lifts a held-out reasoning suite while the
  symbolic verifier — kept as periodic ground-truth audit — detects **no** reward-hacking
  (no rise in high-PRM/symbolically-rejected steps). Both conditions, or the row stays Open.
- Watch the coverage trap: domains with no symbolic checker get no label; report per-domain
  label coverage so a "good" PRM number that only reflects checkable domains is visible.

### 3. W3 — provenance-weighted training + influence

Ledger row: `w3-provenance-weighting-not-validated-vs-loo`.

- Feed the tool's per-example weights + curriculum order into the MLX LoRA run (weight the
  loss / order the data). The influence proxy is model-free; upgrade it to real TracIn or an
  influence-function backend if you wire one.
- **Acceptance gate:** provenance-weighted training **beats uniform weighting** on a held-out
  suite; the influence method's top-implicated rows **agree with leave-one-out retraining** on
  a small slice; and output-diversity metrics show **no register collapse** (the objection).
  All three, with artifacts, or the row stays Open.

### 4. W4 — adversarial gate self-play

Ledger row: `w4-adversarial-selfplay-single-round-dry`.

- Supply a live `model_gate(prompt) -> {completion, fabricated, passed_gate,
  reference_abstain}` (real model + the repo's gate) and run **≥3 rounds**, appending mined
  fabricate-and-pass negatives to the DPO file between rounds and re-training.
- **Acceptance gate:** the base model's **fabricate-and-pass rate drops** on a held-out
  adversarial set across rounds, while proposer **novelty stays above a floor** (no adversary
  collapse — the tool already dedups; track novelty per round). Both, or the row stays Open.

### 5. W5 — probe-as-loss (LAST, highest risk — do not skip the audit)

Ledger row: `w5-probe-as-loss-not-attempted-goodhart-unproven`.

- Reuse the real hidden-state featurizer from W1. Then, and only then, use the truthfulness
  probe as an auxiliary training loss.
- **Acceptance gate (all four):** the probe-as-loss run improves the **held-out AUDIT probe's**
  accuracy (not just the loss probe's); `goodhartGap ≤ 0.15`; a **from-scratch audit probe
  trained AFTER the run still separates** honest/deceptive (the signal was not obfuscated); and
  the loss-probe is treated as BURNED for eval. If the audit probe does not move while the loss
  probe does, that is the Goodhart failure — the row stays Open and you report it as a negative
  result, which is a valuable finding, not a defeat.

---

## Report back to the advisor (this format)

1. **Baseline:** git HEAD + `31 passed` confirmation (or the red diff).
2. **Per tool attempted (W2 first):** what you wired, the exact command, the number you got,
   and **whether its acceptance gate was met** — yes → ledger row flipped (artifact + sha256
   path); no → row stays Open + the number + why it fell short.
3. **Any real binding defect found** (like the advisor's `verify_derivation` dict/verdict fix
   or the mock-provider hazard): file, symbol, the one-line fix.
4. **Negative results are results.** A probe-as-loss that gets gamed, a calibration loss that
   trades accuracy for hedging — report them plainly; they retire a hypothesis honestly.
5. **What you did NOT run** and why (no GPU, no key, out of scope), so the advisor knows the
   true coverage.

Do not `git commit` unless the user asks — leave the files staged/untracked as the advisor did.