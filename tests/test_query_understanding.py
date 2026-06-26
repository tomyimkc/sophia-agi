# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the deterministic query-understanding layer (no API key, no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import query_understanding as qu  # noqa: E402


def test_normalize_lowercases_and_trims() -> None:
    assert qu.normalize("  What IS  the Dao De Jing??  ") == "what is the dao de jing"
    assert qu.normalize("") == ""


def test_language_detection() -> None:
    assert qu.detect_language("who wrote this") == "en"
    assert qu.detect_language("道德经的作者") == "zh"
    assert qu.detect_language("Dao 道德经") == "mixed"
    assert qu.detect_language("12345 ???") == "other"


def test_intent_rules_bilingual() -> None:
    assert qu.classify_intent("What is the Dao De Jing?") == "definition"
    assert qu.classify_intent("什么是道德经") == "definition"
    assert qu.classify_intent("Compare Plato and Aristotle") == "comparison"
    assert qu.classify_intent("比较道家和儒家") == "comparison"
    assert qu.classify_intent("when was it written") == "temporal"
    assert qu.classify_intent("official website of the project") == "navigational"
    assert qu.classify_intent("who wrote war and peace") == "factoid"


def test_comparison_decomposes_to_subqueries() -> None:
    a = qu.analyze("Compare Plato and Aristotle on virtue")
    assert a.is_multi_hop
    assert a.sub_queries[0] == "plato"
    assert "aristotle" in a.sub_queries[1]


def test_cjk_comparison_decomposes_without_whitespace() -> None:
    a = qu.analyze("比较道家和儒家")
    assert a.sub_queries == ["道家", "儒家"]
    assert a.is_multi_hop


def test_factoid_with_and_stays_atomic() -> None:
    # "War and Peace" must NOT be split — only comparison/diff questions fan out.
    a = qu.analyze("Who wrote War and Peace?")
    assert a.sub_queries == ["who wrote war and peace"]
    assert not a.is_multi_hop


def test_proper_name_expansion_drops_leading_command_word() -> None:
    a = qu.analyze("Compare Plato and Aristotle")
    # "Compare Plato" must not leak in as a bogus expansion term.
    assert "compare plato" not in a.expansions
    assert "plato compare" not in a.expansions


def test_alias_expansion_widens_recall() -> None:
    a = qu.analyze("Who wrote War and Peace by Leo Tolstoy?")
    # a surname surface form is recovered from the full name (ordering forms count)
    assert any("tolstoy" in e for e in a.expansions)


def test_analyze_is_deterministic() -> None:
    q = "Compare Plato and Aristotle on the soul"
    assert qu.analyze(q).to_dict() == qu.analyze(q).to_dict()


def test_search_terms_appends_expansions() -> None:
    a = qu.analyze("Who wrote War and Peace by Leo Tolstoy?")
    terms = a.search_terms()
    assert terms.startswith(a.normalized)
    assert "tolstoy" in terms


def test_llm_rewrite_failure_is_swallowed() -> None:
    class _Boom:
        def generate(self, *_a, **_k):  # noqa: ANN001
            raise RuntimeError("no network")

    # client present but failing → deterministic result unchanged, no raise.
    a = qu.analyze("what is the dao de jing", client=_Boom())
    assert a.intent == "definition"
