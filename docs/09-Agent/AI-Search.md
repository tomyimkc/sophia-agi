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
agent/ai_search.py   fuse across sub-queries (RRF) ──► dedup (agent/dedup.py)
  │                                                ──► agent/rerank.py (final BM25-lite order)
  ▼
SearchResult { AnalyzedQuery plan, ranked SourceChunks }
```

> **Where this is heading.** Wired onto Sophia's belief graph + grounded gate + graded
> abstention, this pipeline becomes a *verifiable perception organ* — see
> [Search-as-AGI-Substrate.md](Search-as-AGI-Substrate.md).

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

### Near-duplicate collapse — `agent/dedup.py`
Word-shingle Jaccard clustering collapses genuine duplicates (chunk overlap, `r0`/`r1` teacher
variants) in the candidate pool, keyed on chunk **body** so variant titles don't defeat it.
Opt-in on `retrieve_hybrid(dedupe=True)`, on by default in `ai_search`. Honest bound: it
improves result *diversity*; it does not merge distinct-but-related records (that's a
ranking/field problem), so on the gold-record metric its delta is ~zero here — a finding the
eval's `hybrid_dedup` ablation makes explicit.

### Pluggable embedders — `agent/embedding_backends.py`
The seam where a new embedder — notably a learned **multilingual / multimodal** model (the JD's
"任何语言、任何模态") — registers under a backend id and is picked up by
`retrieval.embed_query_for_index` with no change to the retrieval path. Built-ins:
`local-hash-v1`, `gemini`. Shipping learned weights is out of scope for the offline CI; this is
the contract they plug into.

### Scale-out serving — `services/ann_serving/` + `agent/ann_client.py`
A dependency-free Rust core with **flat (exact)**, **single-layer NSW**, and **multi-layer
HNSW** cosine indexes; HNSW lifts recall over NSW at equal `ef` (benchmarked). `agent/ann_client.py`
drives the `serve` binary as a fail-soft subprocess (falls back to the Python vector path when
the binary/export are absent), so the dense view can be served by the Rust core while Python
keeps understanding, fusion, rerank, and provenance.

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
