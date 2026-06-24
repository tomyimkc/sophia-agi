# Known Limitations: "LLM-as-control-flow, knowledge-in-the-graph"

Companion to [continual-learning-non-parametric.md](continual-learning-non-parametric.md).
This is a deliberate failure-ledger entry, in the spirit of Sophia's public-failure
discipline: before building the integrated system further, name what it structurally
**cannot** do.

## The design under scrutiny

> A **frozen LLM as pure control flow** — it routes, retrieves, composes, and renders
> language — while **all knowledge lives in the OKF / wiki / Obsidian belief graph,
> gated by Sophia.** The LLM holds no facts; the graph holds no reasoning.

## Limitations (deepest first)

1. **The control/knowledge boundary leaks and cannot be sealed.** To decide *what to
   retrieve*, the controller must already understand the query — entities, relevance,
   semantics, plausibility. That interpretive prior **is** parametric knowledge. You can
   externalize declarative *facts*, but not the latent world-model needed to know which
   facts matter. Retrieval recall is upper-bounded by the controller's parametric prior —
   the very thing the design tries to remove.

2. **Coverage ceiling.** The graph contains only what was deliberately authored *with
   sources* and passed the fail-closed gate (`agent/gate.py`). It is necessarily tiny
   next to a weight model's trillions of tokens: high-precision, low-recall by
   construction. Brutal cold-start; value only appears after dense, costly curation.

3. **Provenance is not truth.** The gate verifies a claim *traces to a source*, not that
   the source is correct. Well-cited falsehoods pass. `propagate_confidence`'s
   min-over-chain cannot detect a citation ring (A←B←A) laundering confidence. You get
   auditability, not correctness — easy to conflate.

4. **Hallucination moves, it doesn't die.** Even with perfect facts the LLM composes the
   *connective tissue* and can fabricate relationships, weights, or inference steps. The
   gate cannot check a **novel synthesis that is in no source** — forcing a lose-lose:
   abstain on everything new (restate-only), or pass unverified inference (the original
   risk). This is the [ripple-effect](https://aclanthology.org/2024.tacl-1.16.pdf) problem
   reincarnated.

5. **Symbolic brittleness.** Pages are discrete symbols; tacit/procedural knowledge
   (style, skill), fuzzy/graded/statistical knowledge, and quantitative/computational
   domains resist the OKF schema (`okf/schema.py` is shaped for humanities provenance).
   Frame/qualification problems and entity-resolution drift degrade as the graph grows.

6. **Generalization loss.** Distributed representations cause interference (forgetting)
   *and* give soft generalization — the same mechanism. Moving knowledge to discrete
   symbols cures interference but forfeits fluid analogy/interpolation. You trade
   *forgetting* for *rigidity*.

7. **Consistency cost scales badly.** Minimal-change belief revision (AGM) is hard;
   classic Truth-Maintenance Systems did not scale. Contradiction accumulation, cascade
   cost, and **temporal rot** ("Pluto is a planet" was true) grow with graph density.

8. **The controller is the new single point of failure + attack surface.** Retrieved
   wiki pages are untrusted input flowing into the controller's context → prompt
   injection (`agent/untrusted.py`, `agent/public_sanitize.py` exist precisely for this).
   And the routing logic stays an opaque neural net: knowledge is auditable, *decisions
   about knowledge* are not.

9. **Cost, latency, and faithfulness verification.** Retrieval + graph ops + gate +
   multi-round LLM calls are far heavier than one forward pass, each hop a failure point.
   And it is hard to prove the controller *used* the retrieved page rather than answering
   from parametric memory and citing post-hoc (cf. `agent/legal_faithfulness.py`).

## The honest reframing

Not "LLM = control, graph = knowledge," but **two knowledge stores with different
guarantees**:

- **Weights** — broad, fuzzy, generalizing, *unverifiable*; also the interpretive prior
  that drives retrieval.
- **Graph** — narrow, discrete, *auditable, gated*.

The graph's job is not to *replace* the weights; it is the **authoritative override** for
the slice of knowledge where fabrication is unacceptable. The LLM arbitrates. This keeps
every win of Experiments 1–4 while staying honest about limits 1–9.

**Wins where it is strong:** high-stakes, curated, attribution-sensitive, auditable,
latency-tolerant domains (humanities provenance, legal citations, governance gating) —
where precision/abstention is a feature.
**Structurally cannot win:** open-domain breadth, tacit/procedural/quantitative
knowledge, novel synthesis, low latency, or removing parametric knowledge (limit 1).

---

## Recommended next step: an integrated, benchmarkable harness (CPQA)

To turn the four isolated modules into one measurable system, build a **Continual
Provenance QA (CPQA)** benchmark that runs the full dual-store loop end to end and emits
a report in the repo's existing format
(`agi-proof/benchmark-results/*.public-report.json`, `candidateOnly`, `validated:false`).

**Episodes** (`eval/continual_qa/episodes_v1.jsonl`) — each episode declares:
`{ learn: [OKF pages], retract: [sources], queries: [{q, expectedId|abstain, type}] }`.
Episodes arrive as a *stream* (the no-task-boundary setting the report demands).

**Two systems under test, same questions:**
- `parametric_baseline` — knowledge frozen at episode 0; cannot ingest later pages, so it
  must fail/forget post-episode-0 facts. Stands in for a weight model without retraining.
- `graph_backed` — ingests via `belief_revision_policy` (conflicts) and `Unlearner`
  (retractions); answers only from the **grounded** `belief_state`; abstains (fail-closed)
  when a fact is ungrounded or retracted.

**Metrics (all deterministic, offline, CI-able):**
- accuracy per episode and overall;
- retention matrix + `forgottenGroundedClaims` (from `agent/continual_retention.py`);
- backward transfer;
- **fabrication rate** — answers asserting an ungrounded fact (target: 0 for graph_backed);
- **abstention correctness** — abstains exactly on retracted/ungrounded queries.

**Why this is the right next step:** it is the **sequential-retention protocol the source
report said most pipelines lack**, it integrates all four experiments into one artifact,
it contrasts against a baseline a reviewer understands (the weight model), and it slots
into `tools/run_*_benchmark.py` + `provenance_bench` scoring with no new infra. The
controller can start deterministic (retrieval-only) so the benchmark proves the
*knowledge substrate* first; an LLM-controller mode plugs in later behind the same harness.

**Pre-registered honesty:** `level3Evidence:false` / `validated:false` until it clears the
full no-overclaim gate (≥2 judge families, κ ≥ 0.40, ≥3 runs, CIs) per RESULTS.md. The
baseline must be a fair weight-model analogue, not a strawman.
