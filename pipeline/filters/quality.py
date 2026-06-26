# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Composed Dolma-style document tagger: C4 cleaning → Gopher quality (pure stdlib).

`tag_document` returns a deterministic attributes dict + a fail-closed keep
decision (keep only if BOTH C4 and Gopher pass). `filter_corpus` applies it to a
collection and returns kept docs + drop-reason statistics — the reproducible
quality-filter stage of a FineWeb/Dolma corpus build.
"""
from __future__ import annotations

from pipeline.filters import c4, gopher


def tag_document(text: str, *, min_sentences: int = 5) -> dict:
    """Compute the full quality attributes + fail-closed keep decision for a doc."""
    c4_keep, c4_fails, cleaned = c4.keep(text, min_sentences=min_sentences)
    g_keep, g_fails = gopher.keep(cleaned)
    failed = [f"c4:{f}" for f in c4_fails] + [f"gopher:{f}" for f in g_fails]
    return {
        "keep": bool(c4_keep and g_keep),
        "failed": failed,
        "gopher": gopher.signals(cleaned),
        "cleanedChars": len(cleaned),
        "rawChars": len(text),
    }


def filter_corpus(docs: "list[str]", *, min_sentences: int = 5) -> dict:
    """Apply the tagger to a corpus; return kept docs + drop-reason histogram."""
    kept = []
    reasons: dict[str, int] = {}
    tagged = []
    for d in docs:
        t = tag_document(d, min_sentences=min_sentences)
        tagged.append(t)
        if t["keep"]:
            kept.append(d)
        else:
            for r in t["failed"]:
                reasons[r] = reasons.get(r, 0) + 1
    n = len(docs)
    return {
        "n": n,
        "kept": len(kept),
        "dropped": n - len(kept),
        "keepRate": round(len(kept) / n, 6) if n else 0.0,
        "dropReasons": dict(sorted(reasons.items())),
        "keptDocs": kept,
        "tagged": tagged,
    }
