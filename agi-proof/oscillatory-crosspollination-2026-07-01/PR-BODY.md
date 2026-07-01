## Summary

Five drop-in, fail-closed, unit-tested tools that bring **convergence/coherence** — the signal
`unconv-ai/Un-0` uses to *generate* images from coupled-oscillator (Kuramoto) dynamics — into
sophia's epistemics stack as a **verification and abstention** signal. sophia today measures
trust with rules and verifiers; this adds the one axis its stack is silent on: *how coherently
an answer settles.*

**Instruments, not results.** No tool updates model weights; the learned / hidden-state step is a
documented seam. Every output carries `candidateOnly:true`, `level3Evidence:false`,
`canClaimAGI:false`. The five failure-ledger rows stay **Open** until their acceptance gates are met.

### Tools (`tools/`)
| File | Direction | Binds to |
|---|---|---|
| `oscillator_core.py` | shared Kuramoto engine | numpy |
| `consensus_gate_oscillator.py` | **O1** order-parameter *r* as a self-consistency gate | `agent.calibration`, `agent.selective_risk` |
| `energy_verifier_head.py` | **O2** (flagship) energy verifier; min-energy = Best-of-N | `agent.activation_probes`, `agent.calibration` |
| `fixedpoint_stability_gate.py` | **O3** DEQ-style claim-vs-evidence stability | `oscillator_core`, `agent.selective_risk` |
| `adaptive_compute_convergence.py` | **O4** convergence-driven adaptive test-time compute | `oscillator_core`, `agent.calibration` |
| `oscillator_substrate_sim.py` | **O5** (SIMULATION ONLY) oscillator-Ising relaxation | numpy |

## What each tool proves offline (synthetic fixtures)
- **O1**: on paraphrased-correct vs divergent-wrong samples, Kuramoto *r* beats the majority-
  agreement gate (paired AURC CI excludes 0); on tied signals reports `inconclusive`, never a
  fabricated win; order-invariant.
- **O2**: accepted (answer,evidence) pairs get lower energy than rejected; Best-of-N selection
  1.0; held-out-**domain** `goodhartGap` audited; hidden-state featurizer seam reported unimplemented.
- **O3**: supported claims have lower fixed-point residual than unsupported; reports a data-driven
  `suggestedThreshold`; conservative fail-closed default.
- **O4**: easy queries stop at k≈2, hard/disjoint queries run to k_max and emit an abstain signal;
  ~57% samples saved at no AURC cost.
- **O5**: substrate relaxation reproduces the digital argmin-energy decision with a positive
  annealing gap; `simulationOnly:true`, `hardwareClaim:false` — **not hardware, not LLM-scale.**

## Tests
```
PYTHONPATH=. python -m pytest \
  tests/test_oscillator_core.py tests/test_consensus_gate_oscillator.py \
  tests/test_energy_verifier_head.py tests/test_fixedpoint_stability_gate.py \
  tests/test_adaptive_compute_convergence.py tests/test_oscillator_substrate_sim.py -q
# -> 30 passed. Combined with the WS-A–D + W1–W5 packages: 88 passed, 0 skipped.
```

## Design invariants
- **Fail closed.** No backend / degenerate labels / empty input → environment artifact (`ok:false`)
  or non-zero exit — never a fabricated metric or win.
- **A run is not a result.** The learned/hidden-state step (`build_hidden_state_featurizer`) and
  live wiring are documented seams; see `CONTINUATION-PROMPT.md` for per-tool acceptance gates.
- **`hash_embed` is a lexical stand-in**, the same kind of seam as the probe featurizer; a semantic
  embedder is the drop-in that makes the coherence/residual geometry meaningful (`hashEmbedSeam:true`).
- **O5 does not claim hardware.** It is a software simulation to make the O2→O5 bridge concrete.

## Scope / provenance
- Adds only new files under `tools/`, `tests/`, and `agi-proof/`. **No existing tracked file
  modified**; no personal notes / vault included.
- Prospectus with the full literature survey (AKOrN ICLR 2025 Oral, EBT, DEQ, Neural ODEs, CTM,
  GASPnet, oscillator Ising machines) and the strongest objection to each direction:
  `agi-proof/oscillatory-crosspollination-2026-07-01/Sophia-Oscillatory-CrossPollination.md`.
- Citations give title + venue as surfaced by search; arXiv IDs only where explicitly seen —
  confirm before citing in a paper.