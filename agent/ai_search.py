# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""End-to-end AI-search pipeline: understand → recall (hybrid) → fuse → rerank.

This is the coherent "AI 搜索算法" surface the JD describes, assembled from Sophia's parts:

    query → query_understanding.analyze       (normalize / intent / decompose / expand)
          → per sub-query: hybrid_retrieval     (dense + sparse RRF over the committed index)
          → reciprocal_rank_fusion across sub-queries   (multi-hop merge)
          → rerank.rerank_chunks                (BM25-lite final ordering)
          → SearchResult                        (chunks + the analyzed plan, for explainability)

Every stage is deterministic, offline, and CPU-only (the committed `local-hash-v1` embedder
needs no API key); an optional ``client`` only *adds* HyDE rewrites and an LLM rerank on top
of the deterministic result, never replaces it. The result keeps the :class:`AnalyzedQuery`
so badcase analysis (see ``tools/eval_search_quality.py``) can attribute a miss to a stage —
intent, decomposition, recall, or ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.hybrid_retrieval import reciprocal_rank_fusion, retrieve_hybrid
from agent.query_understanding import AnalyzedQuery, analyze
from agent.retrieval import SourceChunk


@dataclass
class SearchResult:
    query: AnalyzedQuery
    chunks: list[SourceChunk]

    def to_dict(self) -> dict:
        return {
            "query": self.query.to_dict(),
            "chunks": [
                {"path": c.path, "title": c.title, "score": c.score, "excerpt": c.excerpt[:300]}
                for c in self.chunks
            ],
        }


def _chunk_id(c: SourceChunk) -> str:
    return f"{c.path}::{c.title}"


def search(
    query: str,
    *,
    top_k: int = 8,
    client: Any | None = None,
    over_fetch: int = 30,
    rerank: bool = True,
) -> SearchResult:
    """Run the full pipeline and return a :class:`SearchResult`.

    Multi-hop queries fan out: each sub-query is recalled independently (hybrid), then the
    per-sub-query rankings are fused with RRF so a chunk relevant to *several* hops rises.
    The expansion terms widen the recall string for each sub-query.
    """
    plan = analyze(query, client=client)
    expansion_suffix = (" " + " ".join(plan.expansions)) if plan.expansions else ""

    by_id: dict[str, SourceChunk] = {}
    rankings: list[list[str]] = []
    for sub in plan.sub_queries:
        hits = retrieve_hybrid(sub + expansion_suffix, top_k=over_fetch, over_fetch=over_fetch)
        ranking: list[str] = []
        for h in hits:
            cid = _chunk_id(h)
            by_id.setdefault(cid, h)
            ranking.append(cid)
        if ranking:
            rankings.append(ranking)

    if not rankings:
        return SearchResult(query=plan, chunks=[])

    fused = reciprocal_rank_fusion(rankings)
    ordered = [by_id[cid] for cid, _ in fused]

    if rerank and ordered:
        from agent.rerank import rerank_chunks

        ordered = rerank_chunks(plan.search_terms(), ordered, top_k=max(top_k, len(ordered)))

    return SearchResult(query=plan, chunks=ordered[:top_k])


__all__ = ["SearchResult", "search"]
