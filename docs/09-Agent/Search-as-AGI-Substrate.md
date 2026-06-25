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

### 4. A verification gate between retrieval and use
Sophia re-verifies served output (`sophia_mcp/gateway_wiring.verify_output`) and scores citation
faithfulness (`agent/rerank.citation_faithfulness`). Wiring those onto the search path means
retrieved evidence is **checked before it is allowed to ground a claim** — "verification over
generation" applied to perception, so "retrieved" only becomes "believed" after it survives a
check.

### 5. A badcase flywheel → metacognitive self-correction
The search-quality eval (`tools/eval_search_quality.py`) emits a badcase taxonomy
(`lexical_gap` / `semantic_gap` / `tied_burial` / `absent_from_pool`); the fact-check flywheel
(`agent/fact_check_flywheel.py`) can turn each labeled failure into a corrective signal for the
index/embedder. An AGI that **measures its own perceptual errors and closes them** is exhibiting
the operational core of metacognition — knowing what it doesn't know, and fixing it.

## Why this is an AGI-*key* feature, not just better search

Generality in an agent is bounded by the trustworthiness of its inputs. A planner, a coder, a
scientist-agent all inherit the failure modes of their perception. PGVS makes perception
**checkable**: grounded, calibrated, verified, and self-correcting. That is the difference
between an agent that is occasionally right and one that *knows when it is right* — the property
Sophia's charter exists to build, expressed as the organ the DeepSeek charter calls primary.

## Roadmap — from what's shipped to the organ

| Step | From → To | Where |
|------|-----------|-------|
| Ground | `SearchResult` → belief-graph claim nodes with provenance edges | `agent/ai_search.py` → `okf/graph.py` |
| Verify | search path → grounded gate + citation faithfulness before use | `agent/grounded_gate.py`, `agent/rerank.py` |
| Abstain | search path → graded answer/hedge/abstain on low/conflicting confidence | `agent/graded_decision.py`, `agent/grounded_confidence.py` |
| Self-correct | eval badcases → flywheel → index/embedder updates | `tools/eval_search_quality.py` → `agent/fact_check_flywheel.py` |
| Perceive widely | learned multilingual/**multimodal** embedder via the registry | `agent/embedding_backends.py` |
| Serve at scale | Rust HNSW dense view via the bridge → sharding/RDMA | `services/ann_serving/`, `agent/ann_client.py` |

## Shipped foundation (this work)

Understand → hybrid recall (dense+sparse RRF) → dedup → multi-hop fusion → rerank
(`agent/ai_search.py`), a graded search-quality eval体系 with a badcase taxonomy, a pluggable
embedder registry (the multilingual/multimodal seam), and a Rust HNSW serving core with a
Python bridge. See [AI-Search.md](AI-Search.md). Honest status: the *grounding/verification/
abstention/flywheel* wiring above is the next, highest-leverage step — it is what converts a
strong search pipeline into Sophia's perception organ.
