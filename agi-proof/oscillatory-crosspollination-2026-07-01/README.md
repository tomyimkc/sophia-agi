# Oscillatory / dynamical-systems cross-pollination (O1–O5)

Five drop-in, fail-closed, unit-tested tools that bring **convergence/coherence** — the signal
`unconv-ai/Un-0` uses to *generate* images from coupled-oscillator (Kuramoto) dynamics — into
sophia's epistemics stack as a *verification and abstention* signal. Full rationale, the
literature survey (AKOrN ICLR 2025 Oral, EBT, DEQ, Neural ODEs, CTM, GASPnet, oscillator Ising
machines), and the strongest objection to each direction: `out/Sophia-Oscillatory-CrossPollination.md`.

**These are instruments, not results.** No tool updates model weights; the learned/hidden-state
step is a documented seam. Every output carries `candidateOnly:true`, `level3Evidence:false`,
`canClaimAGI:false`. The five failure-ledger rows stay **Open** until their acceptance gates are met.

## Tools (`tools/`)

| File | Direction | Binds to (real repo symbols) |
|---|---|---|
| `oscillator_core.py` | shared Kuramoto engine (`consensus_r`, `run_kuramoto`, `order_parameter`, `hash_embed` seam) | numpy only |
| `consensus_gate_oscillator.py` | **O1** order-parameter *r* as a self-consistency confidence gate | `agent.calibration.self_consistency/calibration_report`, `agent.selective_risk.aurc/paired_aurc_delta_ci` |
| `energy_verifier_head.py` | **O2** (flagship) energy-based verifier; min-energy = Best-of-N + Goodhart audit | `agent.activation_probes` (probe + `build_hidden_state_featurizer` seam), `agent.calibration` |
| `fixedpoint_stability_gate.py` | **O3** DEQ-style fixed-point stability of claim-vs-evidence | `oscillator_core`, `agent.selective_risk.aurc` |
| `adaptive_compute_convergence.py` | **O4** convergence-driven adaptive test-time compute | `oscillator_core`, `agent.calibration`, `agent.selective_risk` |
| `oscillator_substrate_sim.py` | **O5** (SIMULATION ONLY) energy verification as oscillator-Ising relaxation | numpy only |

**O5 is a horizon bet, not a build.** It is a *software simulation* of an idealized
coupled-oscillator relaxation that reproduces O2's digital accept/abstain decision. It is **not
hardware, not an energy measurement, and not an LLM-scale claim** (`simulationOnly:true`,
`hardwareClaim:false`). It exists to make the O2→O5 bridge concrete, nothing more.

## The `hash_embed` seam

O1/O3/O4 embed text with `oscillator_core.hash_embed` — a token-hashing stand-in whose geometry
captures lexical overlap, not deep meaning. This is the **same kind of documented seam** as
`agent.activation_probes.build_hidden_state_featurizer`. Dropping in a real sentence embedder is
what makes the coherence/residual geometry semantic; every tool surfaces `hashEmbedSeam:true`.

## Tests

```
PYTHONPATH=. python -m pytest \
  tests/test_oscillator_core.py \
  tests/test_consensus_gate_oscillator.py \
  tests/test_energy_verifier_head.py \
  tests/test_fixedpoint_stability_gate.py \
  tests/test_adaptive_compute_convergence.py \
  tests/test_oscillator_substrate_sim.py -q
# -> 30 passed
```

(`tests/` holds the full repo suite — run the six files by name, not a bare `pytest tests/`.)
Combined with the earlier WS-A–D + W1–W5 packages: **88 passed, 0 skipped.**

## What each tool proves offline (on synthetic fixtures)

- **O1**: on paraphrased-correct vs divergent-wrong samples, Kuramoto *r* beats the majority-
  agreement gate — paired AURC CI excludes 0 (`consensus_beats_baseline`). On tied signals it
  honestly reports `inconclusive`, never a fabricated win. Order-invariant (seeded shuffle).
- **O2**: accepted (answer,evidence) pairs get lower energy than rejected; min-energy Best-of-N
  selection = 1.0; held-out-**domain** generalization audited (`goodhartGap`); featurizer seam
  reported unimplemented.
- **O3**: supported claims have lower fixed-point residual than unsupported; reports a
  data-driven `suggestedThreshold` rather than a hardcoded one; conservative fail-closed default.
- **O4**: easy queries stop at k≈2, hard/disjoint queries run to k_max and emit an abstain
  signal; ~57% samples saved at no AURC cost (paired CI contains 0).
- **O5**: substrate relaxation reproduces the digital argmin-energy decision with a positive
  annealing gap; honesty flags always set.

## Recommended sequence

**O1 first** (training-free, cheapest, falsifiable against your validated self-consistency
result) → **O2 flagship** (needs a GPU backend to move past the linear-probe stand-in) → O3/O4
(reuse the grounding + long-horizon loops) → O5 (horizon flag, contingent on O2). See
`CONTINUATION-PROMPT.md` for the take-live plan and per-tool acceptance gates.