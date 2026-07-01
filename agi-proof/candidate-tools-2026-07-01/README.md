# Sophia AGI-proof: five drop-in evidence tools (2026-07-01)

This package implements the five highest-leverage OPEN items from `agi-proof/TODO.md`,
each as a runnable, unit-tested drop-in for the `tomyimkc/sophia-agi` tree. It follows
the repo's discipline: **fail-closed everywhere, no overclaim, heavy deps opt-in,
`canClaimAGI:false`.**

> **Honesty boundary.** These are *instruments*, not results. Every tool was written and
> unit-tested **offline with synthetic fixtures**; **none has been run with a live model
> backend.** No metric, pass-rate, CI, or ARC score is claimed anywhere. The failure-ledger
> entries stay **Open** with the acceptance gate that would close each.

## What's here

| Workstream | File | Closes (TODO / ledger) |
|---|---|---|
| A — Independent hidden pack | `tools/make_independent_hidden_pack.py` | Third-Party Reproduction (TODO 15-20, 22); hidden-fresh-pack independence |
| B — Gated self-training (Thesis T1) | `tools/run_t1_gated_self_training.py` | `rlvr-live-run-not-yet-gated`; verifier-gap self-training w/ transfer arm |
| C — ARC-AGI runner | `tools/run_arc_agi_sophia.py` | External Benchmarks (TODO 11) |
| D — Free wins | `tools/ablation_no_executor.patch.md`, `tools/run_long_horizon_timed.py` | No-executor ablation cell (TODO 2); long-horizon timed runs (TODO 7-10) |

Tests: `tests/` (17 tests, all offline). Ledger rows to paste: `ledger/failure-ledger-additions.md`.
Example reviewer input: `examples/reviewer_pack_input.json`.

## First command a maintainer runs

```bash
# from the repo root, with these files dropped into tools/ tests/ etc.:
python3 -m pytest tests/ -q          # 17 passing, no backend needed
```

## Then, in leverage order

1. **A (this week, no GPU):** an external reviewer authors items and runs
   `python3 tools/make_independent_hidden_pack.py --input <their.json> \
     --schema agi-proof/hidden-reviewer-packs/schema.json --corpus wiki/ \
     --out agi-proof/hidden-reviewer-packs/pack-<rev>-<date>.json`,
   then `tools/run_hidden_eval_sophia.py` on a clean clone. Removes the independence
   caveat that sits on even the validated SimpleQA headline.
2. **B (with a live adapter/GPU):** wire the SFT/DPO step and run one gated round:
   `python3 tools/run_t1_gated_self_training.py --traces <round0.jsonl> \
     --heldout-scored <scored.jsonl> --heldout-shifted <shifted.jsonl> --adapter <spec> \
     --out agi-proof/self-train/round0.public-report.json`.
   Either it lifts the *shifted* held-out (real generalization) or it plateaus at the
   verifier ceiling (the roofline's predicted null) — both are findings.
3. **C (external credibility):** place official ARC tasks, then
   `python3 tools/run_arc_agi_sophia.py --tasks arc-data/evaluation --adapter <spec> \
     --out agi-proof/benchmark-results/arc-agi.public-report.json`.
4. **D (free wins):** apply `tools/ablation_no_executor.patch.md`, run the
   `sophia-full,sophia-no-executor` ablation; and fire one 30-min long-horizon run:
   `python3 tools/run_long_horizon_timed.py --spec <spec.json> --minutes 30 --adapter <spec> \
     --out agi-proof/long-horizon/run-30m.public-report.json --events <events.jsonl>`.

## Binding verification (2026-07-01, branch `feat/realtime-grounding-loop`)

The interface signatures were originally fetched from `main`; this copy was then audited
against the working tree on `feat/realtime-grounding-loop` @ `537279f9`. Result:

| Symbol | Status |
|---|---|
| `agent.gate_reward.reward` / `is_abstention` | OK |
| `agent.continual_plasticity.evaluate_update` + `EvalMetric`/`UpdateCandidate`/`PromotionDecision` | OK |
| `agent.long_horizon.build_ledger` / `run_long_horizon` / `LongHorizonResult` | OK |
| `Ablation` + `ABLATION_MODES` (no `use_executor`) | OK — the WS-D gap is real |
| `agi-proof/hidden-reviewer-packs/schema.json` | OK |
| model adapter | **DRIFTED — fixed** (see below) |

**Defect found and fixed.** The tools originally bound to `agent.model.Model(spec).complete(prompt)`,
which does not exist. The real adapter is `resolve_config(spec) -> ModelConfig` +
`default_client(spec).generate(system, user) -> ModelResult` (and `ModelClient` for the
long-horizon `client=`). WS-B/WS-C `load_generator` and WS-D `_make_client` were rewritten to
the real API and verified to import against the live `agent.model`.

**Mock-provider hazard closed.** `agent.model._auto_provider()` returns `"mock"` when no API key
is present, and mock `.generate()` returns fabricated text with `ok=True`. That would let a
keyless run emit a "score" — the opposite of fail-closed. The adapters now treat `cfg.kind ==
"mock"` as "no real backend → return None" (override with `allow_mock=True` in tests only).
Verified: in a keyless environment `load_generator()` returns `None` and no exception escapes.

All 17 offline tests pass against this tree (`PYTHONPATH=. python3 -m pytest tests/ -q`).

## Design invariants (shared by all five)

- **Fail-closed:** no backend / no dep / missing data → an "environment artifact, not a
  score" report or a non-zero exit; never a fabricated number, never a crash.
- **No overclaim:** every emitted artifact carries `candidateOnly:true`,
  `level3Evidence:false`, `canClaimAGI:false`; public numbers need ≥2 judge families or a
  CI excluding zero.
- **Decontam floor:** eval data passes a shingle/Jaccard guard; an unreadable corpus fails
  closed rather than passing silently.
- **Opt-in heavy deps:** torch / mlx / model adapters are lazy-imported and abstain when
  absent; core stays stdlib.
