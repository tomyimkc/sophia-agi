## Summary

Two drop-in, fail-closed, unit-tested tool packages for the AGI-proof effort. **Every tool is
an instrument, not a result** — none has been run against a live model backend, none claims a
metric, and each package's failure-ledger rows stay **Open** until their acceptance gates are
met. All outputs carry `candidateOnly:true`, `level3Evidence:false`, `canClaimAGI:false`.

### WS-A..D — candidate evidence tools (`agi-proof/candidate-tools-2026-07-01/`)
| File | Purpose |
|---|---|
| `tools/make_independent_hidden_pack.py` | Build an independent hidden reviewer pack with shingle/Jaccard decontamination + third-party provenance stamp |
| `tools/run_t1_gated_self_training.py` | `gate_reward`-filtered self-training loop with a 3-way outcome classifier (heldout_lift / verifier_overfit / reward_hacking) |
| `tools/run_arc_agi_sophia.py` | ARC-AGI tasks through the fail-closed gate; exact-grid scoring; abstains on ungrounded/unparseable |
| `tools/run_long_horizon_timed.py` | Timed long-horizon runner with append-only event log + intervention counter |
| `tools/ablation_no_executor.patch.md` | Reviewed ablation-mode diff (documentation only — **not** applied) |

### W1..W5 — untapped training-signal tools (`agi-proof/untapped-training-2026-07-01/`)
Each converts one of Sophia's measurement signals into a *learning* signal — the repo's
largest unexploited gap (it measures epistemic quality extensively but rarely trains on it).

| File | Thesis | Reuses (verified real symbols) |
|---|---|---|
| `tools/distill_process_reward_model.py` | **W1** verifier-distilled PRM | `agent.step_verifier.verify_derivation`, `agent.activation_probes` |
| `tools/train_calibration_objective.py` | **W2** proper-scoring calibration loss | `agent.calibration`, `agent.abstention_scoring` |
| `tools/provenance_weighted_training.py` | **W3** provenance-weighted training + influence | `agent.source_ranking.rank_source` |
| `tools/adversarial_gate_selfplay.py` | **W4** adversarial gate self-play | `agent.temptation.prompt_fabrication_temptation` |
| `tools/probe_representation_training.py` | **W5** probe-as-loss + Goodhart audit | `agent.activation_probes` |

## Tests

Nine offline suites added under `tests/`. Bound to the real `agent.*` interfaces (with
self-cleaning stubs where a backend is absent). Run:

```
PYTHONPATH=. python3 -m pytest \
  tests/test_make_independent_hidden_pack.py tests/test_run_t1_gated_self_training.py \
  tests/test_run_arc_agi_sophia.py tests/test_ws_d_free_wins.py \
  tests/test_distill_process_reward_model.py tests/test_train_calibration_objective.py \
  tests/test_provenance_weighted_training.py tests/test_adversarial_gate_selfplay.py \
  tests/test_probe_representation_training.py -q
# -> 58 passed
```

This PR also fixes a `sys.modules` stub-pollution leak in the two WS tests that install an
`agent` package stub — they now snapshot-and-restore, so sibling suites import the real
`agent.*` (was silently skipping 20 real-interface checks when run together).

## Design invariants

- **Fail closed.** No backend / degenerate input / import failure → an environment artifact
  (`ok:false`) or a non-zero exit — never a fabricated metric.
- **A run is not a result.** The weight-updating step (MLX/LoRA/RLVR) is a documented
  maintainer seam in each tool; see `CONTINUATION-PROMPT.md` for the per-tool acceptance gates.
- **Real backend only.** `agent.model._auto_provider()` returns `"mock"` (fabricated text,
  `ok=True`) with no API key — live runs must assert `cfg.kind != "mock"`.
- **W1 needs sympy** (else `agent.math_verifier` abstains `sympy_unavailable`).

## Scope / notes

- Adds only new files under `tools/`, `tests/`, and `agi-proof/`. **No existing tracked file
  is modified**; no personal notes / vault included.
- The companion prospectus is `out/Sophia-Untapped-Training-Theses.md` (each thesis with its
  strongest objection); the take-live plan is `CONTINUATION-PROMPT.md`.