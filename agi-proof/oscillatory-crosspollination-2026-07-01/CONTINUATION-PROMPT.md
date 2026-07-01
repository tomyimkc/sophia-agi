# Continuation prompt — take O1–O5 (oscillatory cross-pollination) live

You are continuing work in the `tomyimkc/sophia-agi` repo. A prior session added five
fail-closed, unit-tested INSTRUMENTS under `tools/` (O1–O5) that bring convergence/coherence
signals (inspired by `unconv-ai/Un-0`'s Kuramoto image generator) into sophia's epistemics
stack. Your job is to move them from instruments to measured results — honestly.

## Ground rules (non-negotiable)
1. **A run is not a result.** No tool may claim a metric without a real backend and a real,
   decontaminated, labelled dataset. Keep `candidateOnly:true` until an acceptance gate is met.
2. **Real backend only.** `agent.model._auto_provider()` returns `"mock"` (fabricated text,
   `ok=True`) with no API key. Any live run must `assert cfg.kind != "mock"`.
3. **The `hash_embed` seam is lexical, not semantic.** O1/O3/O4 use `oscillator_core.hash_embed`
   (token hashing). Before trusting any coherence/residual result, swap in a real sentence
   embedder (e.g. the repo's existing embedding path used by `agent.vector_store`) and re-run.
4. **Verify interfaces before editing.** The model adapter is
   `default_client(spec).generate(system, user) -> ModelResult` — NOT `Model(spec).complete(...)`.
5. Add only new files or clearly-scoped edits; never `git add -A`.

## Baseline guard
```
PYTHONPATH=. python -m pytest \
  tests/test_oscillator_core.py tests/test_consensus_gate_oscillator.py \
  tests/test_energy_verifier_head.py tests/test_fixedpoint_stability_gate.py \
  tests/test_adaptive_compute_convergence.py tests/test_oscillator_substrate_sim.py -q
# expect: 30 passed. Full repo package suite (WS + W + O): 88 passed, 0 skipped.
```

## Order of work

**O1 first (cheapest, training-free).** Assemble a real decontaminated SimpleQA-style set where
each row has k sampled model answers + a correctness label. Replace `hash_embed` with a semantic
embedder. Run `tools/consensus_gate_oscillator.py`. Gate: `verdict == consensus_beats_baseline`,
paired 95% CI lower bound > 0, across ≥2 seeds. If it does NOT beat the majority-agreement gate,
that is a real (publishable) negative result — record it, don't bury it.

**O2 flagship (needs GPU).** Implement `agent.activation_probes.build_hidden_state_featurizer`
(the shared seam O2 and W1/W5 all name). Train the energy head on real verifier-labelled
(answer, evidence) pairs from `agent.verified_trace_rlvr` outputs. Gates: calibrated energy
(AUROC/ECE, paired-bootstrap CI excluding 0) AND held-out-**domain** `goodhartGap ≤ 0.15`. Then
wire min-energy Best-of-N as a reward path in `tools/run_rlvr.py` (same lane as the W1 PRM).

**O3.** Host the fixed-point stability check inside `agent/realtime_grounding.py:ingest_one` as
a pre-ingestion gate; validate residual/non-convergence rejection on the real C1 fact pack
(`agent/realtime_benchmark.py`). Gate: F1 beats the current admission arm at matched coverage.

**O4.** Wrap the convergence stopping rule around the live self-consistency sampler in
`agent/long_horizon.py`, bounded by its cooperative `deadline_monotonic`. Gate: adaptive-k
matches fixed-k selective accuracy (paired CI contains 0) while cutting mean samples ≥ 25%.

**O5 (horizon, do not overclaim).** Leave as simulation. Any hardware/energy claim needs a real
oscillator-Ising mapping and a measured energy comparison — out of scope for a software repo.

## Report back
For each of O1–O4: dataset used (+ decontamination method), backend + model id (assert not mock),
the gate metric with its CI, PASS/FAIL vs the acceptance gate, and the exact command. Negative
results are results — report them plainly. For O5: confirm it stayed simulation-only.