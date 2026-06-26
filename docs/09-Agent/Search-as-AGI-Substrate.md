# Search as Sophia's AGI substrate — verifiable, provenance-grounded perception

> *"搜索是 AGI 感知世界的原生感官."* — search is AGI's native organ for perceiving the world.

That framing (from the DeepSeek AI-search charter) and Sophia's own charter — *provenance,
verification, calibration, fail-closed reasoning* — meet at one idea: **an AGI must not just
retrieve the world, it must know whether what it retrieved is true, current, and attributable —
and stay silent when it isn't.** Ordinary search returns ten links; RAG returns plausible text.
Neither *grounds* a belief. Closing that gap is where Sophia's search layer stops being a
retrieval feature and becomes a candidate AGI-key capability: a **verifiable perception organ**.

## The capability: Provenance-Grounded Verifiable Search (PGVS)

Five properties, each already seeded in this repo, compose into one organ that perceives,
grounds, verifies, abstains, and self-corrects:

### 1. Retrieval → *justified belief*, not retrieval → text
Every retrieved chunk already carries provenance (`tradition`, `authorConfidence`,
`doNotAttributeTo` — see `agent/retrieval.SourceChunk`). Flowing those into the belief graph
(`okf/graph.py`, `okf/counterfactual.py`) turns a search hit into a **claim node with lineage**,
so the agent can answer *"why do I believe this?"* and *"what would I conclude if this source
were removed?"*. Perception that carries its own justification is the prerequisite for an agent
whose conclusions are auditable rather than asserted.

### 2. Calibrated abstention as a sensory reflex
The graded answer/hedge/abstain router (`agent/graded_decision.py`) plus provenance-derived
confidence (`agent/grounded_confidence.py`) let search return *"I don't know / sources conflict"*
instead of fabricating. An AGI whose **perception fails closed** — knows the edge of what it
found — is categorically safer than one that hallucinates to fill the gap. This is the
"fail-closed, not fail-open" pillar applied to the senses.

### 3. Query understanding → goal decomposition (the agentic seed)
`agent/query_understanding.py` already decomposes a multi-hop ask into atomic sub-queries that
are recalled independently and fused (`agent/ai_search.py`). That *is* the skeleton of an
agent's plan→gather→fuse→verify loop. Scaled, search becomes the **tool-use loop of an agent**
(the JD's "为 Agent 提供更强大的信息检索工具"): a goal expands into evidence needs, each need is
perceived, and the results are reconciled.

### 4. A verification gate between retrieval and use ✅
`agent/verified_search.py` wires this onto the live path: a *generated* answer is re-checked —
citation faithfulness (`agent/rerank.citation_faithfulness`), the epistemic gate
(`agent/gate.check_response`), and source discipline (an answer that affirmatively attributes a
work to a `doNotAttributeTo` author is rejected, negation-aware) — and **withheld fail-closed**
if it does not pass. So "retrieved" only becomes "served" after the answer itself survives the
gate. "Verification over generation" applied to perception.

### 5. A badcase flywheel → metacognitive self-correction ✅
Hedged/abstained/withheld queries are logged as knowledge gaps (`agent/knowledge_gap_log.py`)
and ranked into an enrichment worklist; `agent/gap_ingest.py` then **materializes the
missing-topic gaps into provenance-skeleton draft stubs** (`none_extant`, `needsReview`, *no
claims*) in the quarantined draft tier. The loop closes: a query Sophia couldn't ground now
creates a routable, auto-abstaining **known-unknown** stub, ready for a sourced fill — so the
corpus scaffolding grows exactly where perception failed, with zero fabrication. The
search-quality eval (`tools/eval_search_quality.py`) supplies the complementary badcase taxonomy
(`lexical_gap` / `semantic_gap` / `tied_burial` / `absent_from_pool`). An AGI that **measures its
own perceptual errors and grows to close them** is exhibiting the operational core of
metacognition — knowing what it doesn't know, and acting to fix it.

## Why this is an AGI-*key* feature, not just better search

Generality in an agent is bounded by the trustworthiness of its inputs. A planner, a coder, a
scientist-agent all inherit the failure modes of their perception. PGVS makes perception
**checkable**: grounded, calibrated, verified, and self-correcting. That is the difference
between an agent that is occasionally right and one that *knows when it is right* — the property
Sophia's charter exists to build, expressed as the organ the DeepSeek charter calls primary.

## Roadmap — from what's shipped to the organ

| Step | From → To | Where | Status |
|------|-----------|-------|--------|
| Ground | search results → OKF belief (entity-link + lineage/laundering/contradicts) | `agent/grounded_search.py` → `okf/graph.belief` | ✅ shipped |
| Calibrate + Abstain | search path → provenance confidence → graded answer/hedge/abstain | `agent/grounded_search.py`, `agent/grounded_confidence.py`, `agent/graded_decision.py` | ✅ shipped |
| Verify | generated answer → citation faithfulness + epistemic gate + source-discipline before serving | `agent/verified_search.py` (`agent/rerank.citation_faithfulness`, `agent/gate.check_response`) | ✅ shipped |
| Self-correct | gaps → draft stubs → **sourced fill from trusted sources** (allowlist + gate) | `agent/gap_ingest.py` → `agent/source_fill.py` (`agent/wiki_librarian.py`, `agent/source_ranking.py`) | ✅ shipped; ⚠️ canon promotion + LLM extraction stay operator-gated |
| Perceive widely | learned **multilingual + multimodal** embedders via the registry | `agent/embedding_st.py` → `agent/embedding_backends.py` | ✅ adapter shipped (opt-in); weights operator-fetched |
| Promote to canon | reviewed draft → consolidated memory tier on human sign-off (re-gated) | `agent/canon_promote.py`, `tools/review_drafts.py` | ✅ shipped (human-gated by design) |
| Serve at scale | Rust HNSW dense view via the bridge → sharding/RDMA | `services/ann_serving/`, `agent/ann_client.py` | ⚠️ single-node shipped; sharding/RDMA pending |

## Shipped foundation (this work)

- **Pipeline.** Understand → hybrid recall (dense+sparse RRF) → dedup → multi-hop fusion →
  rerank (`agent/ai_search.py`). See [AI-Search.md](AI-Search.md).
- **Grounded, calibrated perception.** `agent/grounded_search.py` grounds the top result in the
  OKF belief graph (entity-link → lineage, confidence-laundering, `contradicts`,
  `doNotAttributeTo`), derives a **provenance confidence**, and applies the **answer / hedge /
  abstain** reflex — downgrade-only and fail-closed. Measured discrimination over the OKF wiki
  (`tools/eval_grounded_search.py`, candidate): **weak sources downgraded 100%**, strong sources
  answered ~67% (the rest conservatively hedged).
- **Self-correction loop.** Hedged/abstained queries are logged as knowledge gaps that feed the
  existing frequency-ranked enrichment worklist (`agent/knowledge_gap_log.gap_worklist`) — the
  corpus grows where perception actually failed.
- **Scale & breadth seams.** A graded search-quality eval体系 with a badcase taxonomy, a
  pluggable embedder registry (multilingual/multimodal), and a Rust HNSW serving core with a
  Python bridge.

- **Served-answer verification.** `agent/verified_search.py` runs the *generated* answer through
  citation faithfulness + the epistemic gate + a negation-aware source-discipline check, and
  **withholds it fail-closed** if it does not pass — so "retrieved" only becomes "served" after
  the answer survives the gate.
- **Closed self-correction loop.** `agent/gap_ingest.py` + `tools/close_gap_loop.py` turn logged
  gaps into auto-materialized draft stubs (`none_extant`, no claims) in the quarantined draft
  tier — verified end-to-end: an ungrounded query becomes a stub the same query then routes to
  and abstains on.

- **Sourced fill.** `agent/source_fill.py` + `tools/fill_gap_stubs.py` promote a `none_extant`
  stub into a sourced page by extracting it (via the librarian) from a **trusted source** — two
  fail-closed boundaries: the source must be allowlisted (operator-curated `raw/` dir or an
  authority-ranked id, `agent/source_ranking.py`), and the extracted page must pass the
  provenance gate. A filled page is stamped `provenance: librarian_fill` and keeps `needsReview`
  (sourced, but still awaiting human sign-off before canon).

- **Canon promotion.** `agent/canon_promote.py` + `tools/review_drafts.py` surface `needsReview`
  drafts with their provenance + live gate status, and — on explicit **human sign-off** — elevate
  an approved draft into the consolidated memory tier (re-gated, fail-closed: a draft tampered
  after creation is rejected at the boundary). The hand-authored canonical wiki is never touched.
- **Perceive widely.** `agent/embedding_st.py` registers opt-in **learned** backends —
  `st-multilingual-v1` (cross-lingual meaning) and `clip-multimodal-v1` (text+image in one space)
  — through the registry, so retrieval uses them with no path change. Deliberately not the
  committed default (a learned model needs weights + emits platform-sensitive floats, which would
  break the airgap + deterministic-CI + reproducible-index guarantees); absent the optional
  dependency the registry falls back to the committed offline backend.

Honest status: the full loop — ground · calibrate/abstain · verify · self-correct (gap → stub →
sourced fill → **human-reviewed canon promotion**) · perceive-widely — is wired end-to-end. The
two steps that remain deliberately human/operator-gated, by charter rather than omission, are
(1) the **human sign-off** on canon promotion, and (2) the fill **extraction** + learned-model
**weights**, which need a model the operator supplies; neither can launder past the allowlist +
provenance gate.
