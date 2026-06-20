#!/usr/bin/env python3
"""Tests for token-aware chunking, reranking, and citation faithfulness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import chunking, rerank  # noqa: E402


def test_chunk_covers_full_text_with_overlap() -> None:
    paras = [f"Paragraph {i} about source discipline and provenance reasoning." * 3 for i in range(20)]
    text = "\n\n".join(paras)
    chunks = chunking.chunk_text(text, source_id="doc", max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # stable ids
    assert [c.id for c in chunks] == [f"doc#{i}" for i in range(len(chunks))]
    # no chunk wildly over budget (max_chars = 100*4 = 400, allow overlap slack)
    assert all(c.tokens <= 220 for c in chunks)
    # coverage: first and last paragraphs both appear somewhere
    joined = " ".join(c.text for c in chunks)
    assert "Paragraph 0" in joined and "Paragraph 19" in joined


def test_chunk_short_text_single_chunk() -> None:
    chunks = chunking.chunk_text("short text", source_id="d")
    assert len(chunks) == 1 and chunks[0].id == "d#0"


def test_chunk_empty() -> None:
    assert chunking.chunk_text("", source_id="d") == []


def test_lexical_rerank_orders_by_relevance() -> None:
    docs = [
        "the cat sat on the mat",
        "quantum chromodynamics and gluon fields",
        "a cat and a dog play on the mat",
    ]
    order = rerank.lexical_rerank("cat mat", docs, top_k=2)
    top = [i for i, _ in order]
    assert set(top) == {0, 2}  # both cat/mat docs beat the physics doc
    assert order[0][1] > 0


def test_citation_faithfulness_detects_unsupported() -> None:
    sources = ["The Dao De Jing is attributed to Laozi in the Daoist tradition."]
    grounded = rerank.citation_faithfulness(
        "The Dao De Jing is attributed to Laozi in the Daoist tradition and is a core text.", sources
    )
    assert grounded["groundedFraction"] >= 0.5
    ungrounded = rerank.citation_faithfulness(
        "Quarterly revenue grew forty percent due to enterprise sales expansion overseas.", sources
    )
    assert ungrounded["grounded"] is False
    assert ungrounded["unsupported"]


def test_retrieval_uses_chunking() -> None:
    # _iter_markdown should now return multiple chunks for a long doc
    from agent import retrieval

    readme = ROOT / "README.md"
    items = retrieval._iter_markdown(readme)
    assert len(items) >= 1
    # a long doc yields chunk-labelled titles
    if len(items) > 1:
        assert any("[chunk" in title for title, _ in items)


def main() -> int:
    test_chunk_covers_full_text_with_overlap()
    test_chunk_short_text_single_chunk()
    test_chunk_empty()
    test_lexical_rerank_orders_by_relevance()
    test_citation_faithfulness_detects_unsupported()
    test_retrieval_uses_chunking()
    print("test_chunking_rerank: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
