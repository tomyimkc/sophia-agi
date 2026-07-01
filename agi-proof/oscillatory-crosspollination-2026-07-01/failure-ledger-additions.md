# Failure-ledger additions — oscillatory cross-pollination (O1–O5)

Five Open rows. Each ships a runnable, unit-tested INSTRUMENT; none has been run with a live
model backend or moved past its documented seam. Rows stay Open until their acceptance gate is
met. All artifacts carry `candidateOnly:true`, `level3Evidence:false`, `canClaimAGI:false`.

---

### `o1-consensus-gate-not-validated-vs-self-consistency`
- **Status:** Open
- **Claim boundary:** `tools/consensus_gate_oscillator.py` computes a Kuramoto order-parameter
  confidence and runs a paired-AURC head-to-head vs the repo's `self_consistency` gate. Proven
  on synthetic paraphrase fixtures only; `hash_embed` is a lexical stand-in.
- **Acceptance gate:** on a real decontaminated SimpleQA-style set with sampled traces and a
  semantic embedder, `verdict == consensus_beats_baseline` with the paired 95% CI lower bound
  > 0 — reproduced across ≥2 seeds.

### `o2-energy-verifier-linear-stub-not-hidden-state`
- **Status:** Open
- **Claim boundary:** `tools/energy_verifier_head.py` learns energy = −logit of a compatibility
  probe over `featurize_text` (the documented probe stand-in), with Best-of-N selection and a
  held-out-domain Goodhart audit. `hiddenStateFeaturizerReady` is False.
- **Acceptance gate:** implement `build_hidden_state_featurizer`; train the energy head on real
  verifier-labelled (answer, evidence) pairs; show calibrated energy (AUROC/ECE, paired-bootstrap
  CI excluding 0) AND held-out-**domain** `goodhartGap ≤ 0.15`; wire min-energy Best-of-N as a
  reward path in `tools/run_rlvr.py`.

### `o3-fixedpoint-stability-not-validated-on-c1`
- **Status:** Open
- **Claim boundary:** `tools/fixedpoint_stability_gate.py` iterates an evidence-reconstruction
  fixed point and separates supported from unsupported claims by residual on synthetic data.
- **Acceptance gate:** on the real C1 labelled fact pack (`agent/realtime_benchmark.py`), the
  residual/non-convergence gate rejects unsupported live claims at the admission stage with an
  F1 beating the current admission arm at matched coverage.

### `o4-adaptive-compute-not-wired-into-long-horizon`
- **Status:** Open
- **Claim boundary:** `tools/adaptive_compute_convergence.py` replays a convergence-based
  stopping rule over pre-sampled answers; shows sample savings at no AURC loss offline.
- **Acceptance gate:** wire the stopping rule around the live self-consistency sampler inside
  `agent/long_horizon.py` (bounded by its cooperative deadline); on a real query set, adaptive-k
  matches fixed-k selective accuracy (paired CI contains 0) while cutting mean samples ≥ 25%.

### `o5-oscillator-substrate-simulation-only-no-hardware`
- **Status:** Open (horizon bet)
- **Claim boundary:** `tools/oscillator_substrate_sim.py` is a SOFTWARE SIMULATION of an
  idealized oscillator-Ising relaxation that reproduces O2's digital decision. It is **not
  hardware, not an energy measurement, not LLM-scale** (`simulationOnly:true`).
- **Acceptance gate:** contingent on O2. Any hardware/energy claim requires a real OIM (or
  equivalent) mapping and a measured energy comparison — explicitly out of scope for a software
  repo; this row exists to keep the bet honest, not to be closed here.