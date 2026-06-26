# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hybrid (dense + sparse) retrieval over the committed RAG index.

The index already supports two recall views over the *same* chunk set:

  - **dense** — cosine over committed embeddings (`agent.vector_store.search`), good at
    surface-form generalization (morphology, paraphrase) the exact-token scorer misses;
  - **sparse** — BM25-lite term scoring (`agent.rerank.lexical_rerank`), good at rare/exact
    terms (names, IDs, numbers) that a low-dimensional hash embedding blurs together.

Neither dominates. This module fuses them with **Reciprocal Rank Fusion (RRF)** — the
standard, score-scale-free, parameter-light way to combine rankings: a chunk's fused score
is ``sum 1/(k + rank_i)`` over each list it appears in. RRF needs no score normalization
(dense cosine and BM25 live on different scales) and degrades gracefully when one view is
absent (e.g. no embeddings → sparse-only).

House rules: deterministic, offline, CPU-only — the committed `local-hash-v1` embedder needs
no API key, and BM25-lite is pure Python. Honest bound: at this corpus size fusion runs over
the fully-loaded chunk list (a linear scan), exactly like the existing vector path. The
*fusion layer is index-size-agnostic* — swap the dense view for a FAISS/HNSW ANN backend and
the sparse view for an inverted index, and RRF is unchanged. That ANN/inverted-index serving
core is the architecture-track follow-on, not shipped here.
"""

from __future__ import annotations

from agent.retrieval import SourceChunk, _retrieve_keyword, embed_query_for_index

#: RRF damping constant. 60 is the value from the original Cormack et al. paper and the de
#: facto default; larger flattens the contribution of top ranks, smaller sharpens it.
DEFAULT_RRF_K = 60

#: Default fusion weights (dense, sparse). Dense leads because Sophia's corpus is dominated by
#: near-duplicate teacher examples on which BM25 is high-recall but low-precision (it ranks
#: many lexically-identical-but-wrong chunks alike). Weighting sparse as a *minority vote*
#: lets it promote a chunk the dense view also liked without overriding the cleaner dense
#: ranking.
#:
#: HONEST STATUS (measured, tools/eval_search_quality.py): on the short attribution probes
#: the sparse view is **uninformative** (0 correct hits in top-5), and weighted RRF over a
#: deep sparse list still lets that noise bury genuine dense hits — hybrid recall@5 (≈0.28)
#: falls BELOW pure dense (≈0.52). No weight/depth setting recovers parity on this query
#: type; the sparse lexical signal simply does not help when the gold differs from the query
#: only in surface tokens it shares with many distractors. Hybrid is therefore a
#: **candidate-only** path here, NOT a validated improvement — earlier comments claiming
#: "1.0/0.4 recovers dense parity" were wrong and are corrected. Where hybrid is expected to
#: help is rare-exact-term queries (names/IDs/numbers) a low-dim hash embedding blurs; that
#: is unproven on the current probe set. Tune/justify per corpus before trusting it.
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_SPARSE_WEIGHT = 0.4


def reciprocal_rank_fusion(
    rankings: "list[list[str]]", *, k: int = DEFAULT_RRF_K, weights: "list[float] | None" = None
) -> "list[tuple[str, float]]":
    """Fuse ranked id lists into ``[(id, score)]`` sorted best-first.

    Each ``rankings[i]`` is an ordered list of ids (best first). An id's fused score is the
    weighted sum of ``w_i / (k + rank)`` (1-based rank) over every list it appears in. With
    no ``weights`` every list counts equally (classic RRF). Deterministic: ties break by
    first appearance order across the input lists.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    seq = 0
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            if doc_id not in order:
                order[doc_id] = seq
                seq += 1
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    return fused


def _dense_ranking(query_embedding, chunks, *, top_k: int) -> "list[int]":
    """Indices of ``chunks`` by descending cosine to ``query_embedding`` (embeddings only)."""
    import numpy as np

    if query_embedding is None:
        return []
    scored: list[tuple[float, int]] = []
    for i, c in enumerate(chunks):
        if c.embedding is None:
            continue
        denom = (np.linalg.norm(query_embedding) * np.linalg.norm(c.embedding)) or 1.0
        scored.append((float(np.dot(query_embedding, c.embedding) / denom), i))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [i for _, i in scored[:top_k]]


def _sparse_ranking(query: str, chunks, *, top_k: int) -> "list[int]":
    """Indices of ``chunks`` by descending BM25-lite over ``title + text``."""
    from agent.rerank import lexical_rerank

    docs = [f"{c.title} {c.text}" for c in chunks]
    order = lexical_rerank(query, docs, top_k=top_k)
    return [i for i, score in order if score > 0]


def _to_source_chunk(chunk, score: float) -> SourceChunk:
    excerpt = chunk.text[:1200] + ("..." if len(chunk.text) > 1200 else "")
    return SourceChunk(path=chunk.path, title=chunk.title, excerpt=excerpt, score=round(score, 6))


def hybrid_search(
    query: str,
    chunks,
    *,
    top_k: int = 8,
    query_embedding=None,
    over_fetch: int = 30,
    rrf_k: int = DEFAULT_RRF_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
) -> list[SourceChunk]:
    """Fuse dense + sparse rankings over an in-memory ``chunks`` list (IndexedChunk).

    Over-fetches ``over_fetch`` from each view, fuses by weighted RRF, returns the top
    ``top_k`` as :class:`SourceChunk` carrying the fused score. Sparse-only when no
    embeddings are present (and vice versa) — an absent view simply drops out of the fusion.
    """
    if not chunks:
        return []
    dense = _dense_ranking(query_embedding, chunks, top_k=over_fetch)
    sparse = _sparse_ranking(query, chunks, top_k=over_fetch)
    views = [(dense, dense_weight), (sparse, sparse_weight)]
    active = [(r, w) for r, w in views if r]
    if not active:
        return []
    fused = reciprocal_rank_fusion(
        [[str(i) for i in r] for r, _ in active], k=rrf_k, weights=[w for _, w in active]
    )
    out: list[SourceChunk] = []
    for doc_id, score in fused[:top_k]:
        out.append(_to_source_chunk(chunks[int(doc_id)], score))
    return out


def retrieve_hybrid(
    query: str, *, top_k: int = 8, over_fetch: int = 30, dedupe: bool = False
) -> list[SourceChunk]:
    """Hybrid retrieval entrypoint — loads the committed index, fuses, falls back to keyword.

    Mirrors :func:`agent.retrieval.retrieve`'s contract (same signature shape, same graceful
    keyword fallback) so it is a drop-in higher-recall alternative. With ``dedupe=True`` the
    fused pool is over-fetched and near-duplicates are collapsed (`agent.dedup`) before the
    top-``k`` is taken, so r0/r1 variants and chunk-overlap pairs don't waste result slots.
    """
    try:
        from agent.vector_store import index_dir, load_index

        idir = index_dir()
        indexed = load_index(idir)
        if indexed:
            has_embeddings = indexed[0].embedding is not None
            query_embedding = embed_query_for_index(query, idir, has_embeddings=has_embeddings)
            pool_k = max(top_k, over_fetch) if dedupe else top_k
            hits = hybrid_search(
                query, indexed, top_k=pool_k, query_embedding=query_embedding, over_fetch=over_fetch
            )
            if hits:
                if dedupe:
                    from agent.dedup import dedupe_chunks

                    hits = dedupe_chunks(hits)[:top_k]
                return hits[:top_k]
    except Exception:
        pass
    return _retrieve_keyword(query, top_k=top_k)


__all__ = [
    "DEFAULT_RRF_K",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "retrieve_hybrid",
]
