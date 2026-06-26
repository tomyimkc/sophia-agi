# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for hybrid (dense + sparse RRF) retrieval and the AI-search orchestrator.

Offline & deterministic: committed local-hash index, no API key, no LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import hybrid_retrieval as hr  # noqa: E402


def test_rrf_combines_and_ranks() -> None:
    # 'b' appears high in both lists → must rank first.
    fused = hr.reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])
    ids = [doc for doc, _ in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_weights_bias_toward_weighted_list() -> None:
    # Same item at rank 1 in list 0 and rank 1 in list 1; weighting list 0 up promotes 'x'.
    fused = hr.reciprocal_rank_fusion([["x"], ["y"]], weights=[2.0, 0.5])
    assert fused[0][0] == "x"


def test_rrf_empty() -> None:
    assert hr.reciprocal_rank_fusion([]) == []


def test_hybrid_search_sparse_only_when_no_embeddings() -> None:
    from agent.vector_store import IndexedChunk

    chunks = [
        IndexedChunk("c0", "p0", "Dao De Jing authorship", "The Dao De Jing is attributed to Laozi.", None, "source"),
        IndexedChunk("c1", "p1", "Unrelated", "A note about cooking pasta.", None, "source"),
    ]
    # No query_embedding and no chunk embeddings → sparse-only path, still returns a hit.
    hits = hr.hybrid_search("who wrote the dao de jing", chunks, top_k=2, query_embedding=None)
    assert hits and hits[0].title == "Dao De Jing authorship"


def test_retrieve_hybrid_returns_results_over_committed_index() -> None:
    hits = hr.retrieve_hybrid("who wrote the dao de jing", top_k=5)
    assert hits
    assert all(hasattr(h, "path") and hasattr(h, "score") for h in hits)


def test_ai_search_pipeline_runs_and_carries_plan() -> None:
    from agent.ai_search import search

    result = search("Compare Plato and Aristotle", top_k=5)
    assert result.query.intent == "comparison"
    assert result.query.is_multi_hop
    assert len(result.chunks) <= 5
    d = result.to_dict()
    assert d["query"]["subQueries"] and "chunks" in d
