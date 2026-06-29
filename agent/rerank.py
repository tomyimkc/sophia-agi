# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reranking + citation faithfulness for the RAG layer.

- Deterministic lexical rerank (BM25-lite) so retrieval can over-fetch then
  re-order by query relevance without a model.
- Optional LLM rerank via the unified adapter when a client is supplied.
- Citation faithfulness: does each answer sentence have lexical support in the
  retrieved sources? Turns "cited" into "actually grounded".
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9一-鿿]+", text.lower())


def _bm25_lite(query_tokens: list[str], doc_tokens: list[str], *, avg_len: float, k1: float = 1.5, b: float = 0.75) -> float:
    if not doc_tokens:
        return 0.0
    counts = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        tf = counts.get(term, 0)
        if not tf:
            continue
        score += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1.0, avg_len)))
    return score


def lexical_rerank(query: str, docs: list[str], *, top_k: int | None = None) -> list[tuple[int, float]]:
    """Return (original_index, score) sorted by BM25-lite relevance, best first."""
    q = _tokens(query)
    tokenized = [_tokens(d) for d in docs]
    avg_len = (sum(len(t) for t in tokenized) / len(tokenized)) if tokenized else 1.0
    scored = [(i, _bm25_lite(q, dt, avg_len=avg_len)) for i, dt in enumerate(tokenized)]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:top_k] if top_k else scored


def rerank_chunks(query: str, chunks: list[Any], *, top_k: int = 5, text_attr: str = "excerpt") -> list[Any]:
    """Rerank objects that carry text in ``text_attr`` (e.g. SourceChunk.excerpt)."""
    docs = [str(getattr(c, text_attr, "") or getattr(c, "text", "")) for c in chunks]
    order = lexical_rerank(query, docs, top_k=top_k)
    return [chunks[i] for i, _ in order]


def llm_rerank(query: str, docs: list[str], client: Any, *, top_k: int = 5) -> list[int]:
    """Ask a model to rank documents by relevance; falls back to lexical on failure."""
    listing = "\n".join(f"[{i}] {d[:300]}" for i, d in enumerate(docs))
    system = "You rank passages by relevance. Output ONLY a JSON array of indices, best first."
    result = client.generate(system, f"Query: {query}\n\nPassages:\n{listing}\n\nReturn the top {top_k} indices.")
    if result.ok:
        import json

        match = re.search(r"\[[\d,\s]*\]", result.text)
        if match:
            try:
                idx = [int(i) for i in json.loads(match.group(0)) if isinstance(i, int) and 0 <= i < len(docs)]
                if idx:
                    return idx[:top_k]
            except (json.JSONDecodeError, ValueError):
                # Malformed model output -> fall through to lexical rerank below.
                pass
    return [i for i, _ in lexical_rerank(query, docs, top_k=top_k)]


def citation_faithfulness(answer: str, sources: list[str], *, support_threshold: float = 0.3) -> dict[str, Any]:
    """Fraction of substantive answer sentences with lexical support in sources."""
    source_tokens = [set(_tokens(s)) for s in sources]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", answer) if len(s.strip()) > 25]
    if not sentences:
        return {"grounded": True, "groundedFraction": 1.0, "unsupported": [], "sentenceCount": 0}
    unsupported: list[str] = []
    for sentence in sentences:
        st = set(_tokens(sentence))
        if not st:
            continue
        best = max((len(st & src) / len(st) for src in source_tokens), default=0.0)
        if best < support_threshold:
            unsupported.append(sentence[:120])
    grounded_fraction = 1.0 - len(unsupported) / len(sentences)
    return {
        "grounded": not unsupported,
        "groundedFraction": round(grounded_fraction, 3),
        "unsupported": unsupported,
        "sentenceCount": len(sentences),
    }
