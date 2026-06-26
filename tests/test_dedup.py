# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for near-duplicate collapse (deterministic, offline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dedup  # noqa: E402


def test_jaccard_bounds() -> None:
    a = dedup.shingles("the dao de jing is attributed to laozi")
    assert dedup.jaccard(a, a) == 1.0
    assert dedup.jaccard(a, dedup.shingles("a recipe for tomato pasta sauce")) < 0.2


def test_short_text_falls_back_to_unigrams() -> None:
    # Fewer words than k → unigram bag, so short titles still compare.
    assert dedup.shingles("dao jing", k=4) == frozenset({"dao", "jing"})


def test_dedupe_keeps_first_collapses_rest() -> None:
    items = ["alpha beta gamma delta epsilon", "alpha beta gamma delta epsilon", "x y z w v"]
    kept, dropped = dedupe = dedup.dedupe(items, text_of=lambda s: s)
    assert kept == ["alpha beta gamma delta epsilon", "x y z w v"]
    assert dropped == 1


class _Chunk:
    def __init__(self, title: str, excerpt: str) -> None:
        self.title = title
        self.excerpt = excerpt


def test_dedupe_chunks_collapses_variants_by_body_not_title() -> None:
    # r0/r1 differ only in the title label but share the body → must collapse to one.
    body = "The Dao De Jing is attributed to Laozi by long-standing tradition."
    chunks = [
        _Chunk("Dao De Jing r0", body),
        _Chunk("Dao De Jing r1", body),
        _Chunk("Analects", "The Analects records the sayings of Confucius and disciples."),
    ]
    survivors = dedup.dedupe_chunks(chunks)
    assert [c.title for c in survivors] == ["Dao De Jing r0", "Analects"]


def test_distinct_records_are_not_merged() -> None:
    # Different content with a shared question template must NOT be collapsed.
    chunks = [
        _Chunk("A", "Who wrote the Republic? It is attributed to Plato of Athens."),
        _Chunk("B", "Who wrote the Poetics? It is attributed to Aristotle of Stagira."),
    ]
    assert len(dedup.dedupe_chunks(chunks)) == 2
