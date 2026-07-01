# Failure-ledger additions — untapped training signals (W1–W5)

Paste into `agi-proof/failure-ledger.md`. All five are **Open**: the instrument exists and is
unit-tested, but no live training run has met the acceptance gate. An instrument is not a result.

---

**ID:** `w1-verifier-distilled-prm-not-trained-live`
**Status:** Open
**Claim boundary:** `tools/distill_process_reward_model.py` distills per-step labels from the
fail-closed `verify_derivation` and measures held-out + held-out-domain agreement. It does NOT
train a real PRM on LM hidden states, and does NOT wire the PRM as an RLVR reward.
**Acceptance gate to close:** a PRM trained on real residual-stream features reaches ≥0.80
held-out-DOMAIN agreement with the symbolic oracle AND, used as dense reward in
`tools/run_rlvr.py`, lifts a held-out reasoning suite without the symbolic-verifier audit
detecting reward-hacking. Artifact + checksum under `agi-proof/benchmark-results/`.

---

**ID:** `w2-calibration-objective-not-in-lm-training`
**Status:** Open
**Claim boundary:** `tools/train_calibration_objective.py` lowers ECE on a FROZEN model's
confidences (proper-scoring + asymmetric abstention loss), measured by `agent.calibration`. It
does NOT fine-tune the base LM with this loss.
**Acceptance gate to close:** the loss wired into the MLX/LoRA DPO path lowers ECE on a
held-out, verifier-checkable eval set at matched or better accuracy-at-coverage (no hedging
collapse), across ≥2 seeds on one pinned commit.

---

**ID:** `w3-provenance-weighting-not-validated-vs-loo`
**Status:** Open
**Claim boundary:** `tools/provenance_weighted_training.py` derives loss weights + curriculum
from the real `rank_source`, and provides a model-free influence PROXY. It does NOT run a real
influence function and does NOT fine-tune.
**Acceptance gate to close:** provenance-weighted training beats uniform weighting on a
held-out suite; the influence proxy's top-implicated rows agree with leave-one-out retraining
on a small slice; output-diversity metrics show no register collapse.

---

**ID:** `w4-adversarial-selfplay-single-round-dry`
**Status:** Open
**Claim boundary:** `tools/adversarial_gate_selfplay.py` scores prompts by the real
`prompt_fabrication_temptation`, applies novelty + realism guards, and (with a backend) mines
fabricate-and-pass DPO negatives. Ran DRY only; no live model/gate; no multi-round co-training.
**Acceptance gate to close:** ≥3 self-play rounds with a live model+gate reduce the base
model's fabricate-and-pass rate on a held-out adversarial set, with proposer novelty staying
above a floor (no adversary collapse) across rounds.

---

**ID:** `w5-probe-as-loss-not-attempted-goodhart-unproven`
**Status:** Open
**Claim boundary:** `tools/probe_representation_training.py` proves the Goodhart-AUDIT
methodology (disjoint held-out audit probe, goodhartGap gate) on transparent features. The real
residual-stream featurizer is a stub; the probe-as-loss LM coupling is NOT performed.
**Acceptance gate to close:** with real hidden states, a probe-as-loss run improves the AUDIT
probe's accuracy (not just the loss probe's), goodhartGap ≤ 0.15, and a from-scratch audit
probe trained after the run still separates honest/deceptive — i.e. the signal was not
obfuscated. Until then this direction stays quarantined as the highest-risk of the five.