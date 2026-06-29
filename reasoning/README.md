# `reasoning/` — "thinking as compute" experiments

Falsifiable, offline tests of the [Reasoning-As-Compute](../docs/06-Roadmap/Reasoning-As-Compute.md)
thesis: *treat thinking the way an operator team treats compute — a bounded resource spent
against a measurable ideal, with a fail-closed correctness invariant the efficiency win must
never break.* Each module is pure stdlib, seeded, and runs with no GPU or API keys.

| Module | Thesis feature | Claim tested (all CONFIRMED) |
|---|---|---|
| `deliberation_roofline.py` | #2 budgeting | quality vs. budget is concave; a finite ridge point; the ceiling is the **verifier's** SNR, not compute. MC matches closed form within 0.0019. |
| `reasoning_compiler.py` | #3 compiler/IR | CSE + dead-code elimination cut verification cost ~53% with the grounded conclusion **invariant** (100%); contradictions caught fail-closed, zero false alarms. |
| `memory_hierarchy.py` | #4 memory | a locality-aware tiered policy falls to **8.6%** of flat cost at high locality, with a capacity knee, 100% recall, provenance preserved. |
| `belief_allreduce.py` | #5 collectives | ring/tree reach the same consensus as all-to-all at O(N)/O(N log N) messages; minority beliefs survive (a vote drops them); confidentiality firewall holds. |
| `instinct_gate.py` | #6 instinct | early reflex *re-route* ("change its mind") beats *late* self-correction (0.73 vs 0.55, commit 0.53) **only above a break-even reflex SNR (d′) = 1.0**; below it a trigger-happy reflex hurts; the ko guard bounds re-route to a clean `escalate`. The ceiling is the reflex's ROC, not the policy. See [`Thinking-Chain-Intervention-and-Instinct.md`](../docs/06-Roadmap/Thinking-Chain-Intervention-and-Instinct.md). |
| `instinct_reflex_eval.py` | #6 instinct (measurement) | the go/no-go harness for #6: measures a real reflex's **d′ / AUC** against the belief-revision oracle and checks it clears the break-even bar. Self-consistency disagreement clears d′=1.0 **only when the reasoner is competent** (d′ 0.96→1.74 as competence 0.62→0.95); a no-signal control collapses to AUC≈0.5 (harness manufactures no separation). The harness is the deliverable; the real-model d′ is a gated next step. |

Each module shares one CLI:

```bash
python reasoning/<module>.py --run         # the experiment + a THEORY VERDICT
python reasoning/<module>.py --self-test    # assert the invariants (also in tests/)
python reasoning/<module>.py --run --json   # raw results
```

Saved verdicts live in [`results/`](results/). Tests are in
`tests/test_deliberation_roofline.py`, `tests/test_reasoning_compiler.py`,
`tests/test_memory_hierarchy.py`, `tests/test_belief_allreduce.py`,
`tests/test_instinct_gate.py`, `tests/test_instinct_reflex_eval.py`.

**Honest scope.** These are *models* of the claims, not production wiring — synthetic streams
and planted ground truth, chosen so the hypotheses are falsifiable rather than assumed. Each
maps to a real module it is a candidate to inform (`agent/graded_decision.py`, `okf/graph.py`,
`agent/memory.py`, `agent/sector_council.py`). Not an AGI claim; see `VISION.md`.
