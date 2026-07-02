# 2 · Grounded Context (RAG + Evidence + Memory)

**Role in the master flow.** Assembles the grounded context the model answers from: retrieved
passages, local/web evidence, and prior memory — each with a provenance/trust tag. Ablation flags
`use_kb` (retrieval), `use_evidence` (evidence), `use_memory` (memory).

```mermaid
flowchart TD
    IN([Typed request]) --> RET["RAG retrieval<br/>agent/retrieval.py · rag_pipeline.py<br/>flag: use_kb"]
    IN --> EVI["Local + web evidence<br/>agent/web_evidence.py · live_sources.py<br/>flag: use_evidence"]
    IN --> MEM[("Append-only memory<br/>agent/memory.py · memory/*.jsonl<br/>flag: use_memory")]

    RET --> EMB["Embed & score<br/>agent/rag_embed.py · embedding_backends.py"]
    EMB --> RANK["Trust-rank sources<br/>agent/source_ranking.py<br/>OKF/belief 0.95 → web 0.86 → generic 0.55"]
    EVI --> RTG["Realtime grounding gate<br/>agent/realtime_grounding.py<br/>→ conformal_gate · fact_check_gate"]
    RANK --> PACK
    RTG --> PACK
    MEM --> PACK
    PACK["Compose grounded context<br/>+ context-pack cards<br/>schema/context-pack-card-1.0.0.json<br/>flag: use_context_packing"] --> OUT([To council / answer →])

    RANK -.->|low-trust or no source| ABSTAIN["Flag for abstention<br/>(feeds calibration)"]
```

**Modules:** `agent/retrieval.py`, `rag_pipeline.py`, `rag_embed.py`, `rag_local_embed.py`,
`embedding_backends.py`, `web_evidence.py`, `live_sources.py`, `source_ranking.py`,
`grounded_confidence.py`, `realtime_grounding.py`, `memory.py`.

**Thesis note.** Two traps worth stating in a methods chapter: (1) `rag_local_embed.py` is *also*
hash-based (`local-hash-v1`), so it is not a semantic upgrade over the lexical embedder — confirm the
live backend via `agent.vector_store.embedding_backend_id()`. (2) Source trust rank is deterministic
(`agent/source_ranking.py`), which is what lets provenance become a *weight* on downstream loss (see
the untapped-training W3 direction), not just a display tag.