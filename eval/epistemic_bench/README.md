# Epistemic Bench — Phase 0 of the World-Model & Self-Scaffolding Program

**Status:** candidate infrastructure. `candidateOnly: true`, `level3Evidence: false`,
`canClaimAGI: false`. This is a *measurement harness*, not a capability claim.

Sealed, deterministic, 6-axis benchmark for grounded/abstaining epistemic behaviour. It is the
**build-first** prerequisite for every thesis in
[`docs/06-Roadmap/World-Model-And-Self-Scaffolding-Program.md`](../../docs/06-Roadmap/World-Model-And-Self-Scaffolding-Program.md):
nothing in that program is measurable without it.

## The six axes (each a hard, deterministic check — no LLM judge in the scorer)

| Axis | Measures | Reuses |
|---|---|---|
| **E1** grounding | hallucinated-attribution rate | `agent.gate.check_response`, `agent.verifiers.provenance_faithful` |
| **E2** abstention correctness | abstain precision / recall on the unanswerable split | `agent.gate_reward.is_abstention` |
| **E3** calibration | ECE, risk-coverage AUC | `agent.calibration.calibration_report` |
| **E4** citation faithfulness | cited sources actually support the claim | `agent.verifiers.citation_faithful` |
| **E5** deception robustness | model declines forbidden lineage under injected fabricated / retracted / contradictory sources | case-level `provenance_faithful` |
| **E6** consistency | OKF belief-graph contradiction ledger | `okf.graph.contradiction_ledger` |

E5 deliberately scores the **model's own output**, not the gate's backstop — otherwise the gate
catching a fabricator would mask that the model was fooled. Any axis whose inputs are absent reports
`null` (N/A); the scorer never guesses a pass (fail-closed).

## Run

```bash
# CI fixture — no model needed; asserts a disciplined agent beats a fabricator on E1/E2/E5
python -m eval.epistemic_bench.score --self-test

# Score a model's completions against the sealed cases
python -m eval.epistemic_bench.score \
    --cases eval/epistemic_bench/data/cases.jsonl \
    --completions runs/<model>.jsonl \
    --wiki wiki/                         # enables E6
```

**Completions format** (one JSON object per line):
```json
{"id": "<case id>", "completion": "<model text>", "confidence": 0.8, "correct": true}
```
`confidence` + `correct` are optional and only feed E3 (calibration).

## Splits

`data/cases.jsonl` carries three splits — `answerable`, `unanswerable`, `adversarial`. The seed set
is small and drawn from the attribution register (Dao De Jing / Analects, with forbidden lineages) so
it exercises every axis deterministically. **Grow it before trusting any delta** — `tools/eval_stats.py`
`required_n_for_mde` tells you the N needed to detect a given effect.

## Discipline (non-negotiable, mirrors `RESULTS.md`)

- **Sealed:** the `adversarial` and `unanswerable` splits are held back. No training/synthesis step
  may read them — enforce with `heldout_seal_guard` and a MinHash decontamination check (reuse the
  `pretraining/` passport tooling) between any training corpus and these files.
- **Honest stats:** every rate carries a 95% bootstrap CI (`tools.eval_stats.bootstrap_ci_paired`).
  Deltas between models use paired CIs and `verdict_or_underpowered` — refuse a verdict when the MDE
  exceeds the observed delta.
- **Illustrative → validated:** a number is *illustrative* until it clears ≥2-family judge consensus
  (κ ≥ 0.40 or CI excluding zero) + ≥3 seeds. Only then is it *validated*.

## M0 acceptance gate (OPEN)

Phase 0 is "passed" when the scorer **reproduces the published provenance-delta** (36.1% → 23.6%
hallucinated attribution, CI [5.6%, 19.4%]) on the local 8B baseline within CI — i.e. E1 on the
baseline vs. the grounded variant recovers the known result. Until those two completion sets are run
through this scorer, **M0 is OPEN**. A harness that cannot reproduce a known result cannot certify a
new one.
