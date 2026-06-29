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
| `instinct_fusion.py` | #6 instinct (fusion) | a **2nd independent detector** (real `okf` grounding-closure, catches confident structural errors self-consistency misses) **fused** with self-consistency: neither clears d′=1.0 alone (A 0.87, B 0.97) but the fusion does (d′ 1.86, AUC 0.86) at low correlation (ρ≈−0.22). Pins the law `d′_fused=(d_A+d_B)/√(2+2ρ)` (MC=closed form) — the gain **vanishes as detectors become redundant**. Independence is the whole game. Also exports the 3rd detector `_reflex_B2` (grounding-*completeness*, under-abstention) + `fuse()` (quality-weighted) used by the real-model runner. |
| `instinct_injection.py` | #6 instinct (injection) | the *other* half of the thesis: **edit the chain in place** (backtrack-token / activation-steering) when the reflex fires, vs abandon-and-restart. In-place injection **dominates** re-route (correct 0.66 vs 0.50 at 5× lower cost) — but only at a good steering strength: a **brittleness roofline** (peak at s≈0.5, over-steering s=1.0 collapses to 0.38 as corruption overtakes flips). The inject→reroute **hybrid** wins (0.81). Planted flip/corrupt curves; real white-box steering is the gated next step. |
| `instinct_validation.py` | #6 instinct (rigor) | cross-validates the fusion (LOO-CV weights, offline) + bootstrap CIs. DeepSeek `fused_equal` AUC **0.984 [0.949,1.0]**; LOO-CV qw d′ 5.19→4.84 (small optimism). **Sharp finding:** B/B2 are structural *verifiers* (fire iff answer≠truth ⇒ near-tautological AUC); the only label-free *predictive* reflex is A, and it's weak (AUC 0.63, CI includes chance). If you have the okf graph, verify directly; the frontier is a better label-free reflex. |
| `instinct_labelfree.py` | #6 instinct (frontier) | attacks the frontier §3f named — a better *label-free* reflex than exact self-consistency. Per-element membership **instability** beats exact (AUC **0.668 vs 0.629**) and its CI **excludes chance** (exact's doesn't) — a modest real lift, offline/free from stored samples. Hard limit: **all** agreement signals are anti-predictive on a confident-wrong model (haiku AUC ~0.02) — agreement can't catch consistent errors; that needs model internals. |
| `instinct_endtoend.py` | #6 instinct (outcome) | the payoff: a reflex with a **real measured operating point** (3-detector bus incl. B2) drives the re-route policy. **DeepSeek:** confident-wrong **0.58→0.00**, correct 0.42→0.79. **Claude-haiku (rescued):** still can't do the task (0.02→0.08) but confident-wrong **0.98→0.00**, converted to 0.92 honest **escalation** — fail-closed by reflex. Confident-wrong falls monotonically with detector recall. |

Each module shares one CLI:

```bash
python reasoning/<module>.py --run         # the experiment + a THEORY VERDICT
python reasoning/<module>.py --self-test    # assert the invariants (also in tests/)
python reasoning/<module>.py --run --json   # raw results
```

Saved verdicts live in [`results/`](results/). Tests are in
`tests/test_deliberation_roofline.py`, `tests/test_reasoning_compiler.py`,
`tests/test_memory_hierarchy.py`, `tests/test_belief_allreduce.py`,
`tests/test_instinct_gate.py`, `tests/test_instinct_reflex_eval.py`,
`tests/test_instinct_fusion.py`, `tests/test_instinct_endtoend.py`,
`tests/test_instinct_validation.py`, `tests/test_instinct_injection.py`,
`tests/test_instinct_labelfree.py`.

**Honest scope.** These are *models* of the claims, not production wiring — synthetic streams
and planted ground truth, chosen so the hypotheses are falsifiable rather than assumed. Each
maps to a real module it is a candidate to inform (`agent/graded_decision.py`, `okf/graph.py`,
`agent/memory.py`, `agent/sector_council.py`). Not an AGI claim; see `VISION.md`.
