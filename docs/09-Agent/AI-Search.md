# AI Search pipeline

Sophia's *AI 搜索算法* surface: a deterministic, offline, CPU-only search pipeline assembled
from the existing recall/rerank parts, with a real query-understanding front end and a
hybrid (dense + sparse) recall stage. It is the algorithm-track answer to "让搜索结果更准确、
权威、实时，更适合 LLM 理解与使用" — built to the repo's house rules (no API key required; an
LLM client only *adds* signal, never gates the path).

## Pipeline

```
query
  │  agent/query_understanding.py   normalize · detect language · classify intent
  │                                 · decompose (multi-hop) · expand (aliases + synonyms)
  ▼
sub-queries ──► agent/hybrid_retrieval.py   per sub-query: dense cosine  ⨁  sparse BM25-lite
  │                                          fused by weighted Reciprocal Rank Fusion
  ▼
agent/ai_search.py   fuse across sub-queries (RRF) ──► agent/rerank.py (final BM25-lite order)
  ▼
SearchResult { AnalyzedQuery plan, ranked SourceChunks }
```

Every stage is deterministic. The committed `local-hash-v1` embedder
(`agent/rag_local_embed.py`) makes dense recall work under `SOPHIA_PROFILE=airgap` with no key.

## Stages

### Query understanding — `agent/query_understanding.py`
- **normalize / language** — lowercase + whitespace collapse; en/zh/mixed/other from CJK share.
- **intent** — definition / comparison / temporal / navigational / factoid via small, auditable
  bilingual keyword rules.
- **decompose** — comparison/conjunctive questions fan out to atomic sub-queries
  (`"compare A and B"` → `["A", "B"]`; CJK `比较 A 和 B` splits without whitespace), each
  recalled independently then fused. A plain `"who wrote War and Peace"` stays atomic.
- **expand** — author surface forms (reused from `agent/entity_aliases.py`, incl. cross-lingual
  aliases like *Plato → 柏拉圖*) plus a small curated seed synonym map. Additive only.
- Optional `rewrite_with_llm(query, client)` adds HyDE-style phrasings; `[]` on any failure.

### Hybrid retrieval — `agent/hybrid_retrieval.py`
Dense cosine and sparse BM25-lite over the **same** index, fused by **weighted Reciprocal Rank
Fusion** (scale-free, parameter-light). Default weights dense 1.0 / sparse 0.4 — sparse enters
as a *minority vote* because Sophia's near-duplicate teacher-example corpus makes BM25
high-recall / low-precision (see the eval finding below). The fusion layer is
**index-size-agnostic**: swap the dense view for the Rust ANN core (`services/ann_serving`) or
an HNSW/FAISS backend at scale and RRF is unchanged.

### Orchestration — `agent/ai_search.py`
`search(query, top_k=…, client=None)` runs the whole pipeline and returns a `SearchResult`
that carries the `AnalyzedQuery` plan alongside the ranked chunks — so a miss is attributable
to a stage (intent, decomposition, recall, or ranking).

## Evaluation

`tools/eval_search_quality.py` + `eval/search_quality/` — graded **nDCG@k / recall@k / MRR**
across keyword/vector/hybrid plus a **badcase taxonomy** (`lexical_gap`, `semantic_gap`,
`tied_burial`, `absent_from_pool`). Honest current finding: pure **dense** wins on this corpus;
hybrid beats keyword and closes lexical gaps but doesn't beat dense, because sparse needs
near-duplicate dedup first. The harness *revealing* that — with an explainable, reproducible
signal — is exactly the *搜索质量评估体系* the role asks for.

## Honest bounds

- Embeddings are the committed lexical-semantic **hash** backend (generalises surface form,
  not deep meaning); a learned multilingual embedder is the quality upgrade when a key/GPU is
  available. The query layer and fusion are embedder-agnostic.
- Intent rules and the synonym seed are **hand-authored**, not learned.
- Recall runs over the fully-loaded chunk list (small corpus); the ANN serving core
  (`services/ann_serving`) is the scale path and is not yet wired into the Python dense view.
