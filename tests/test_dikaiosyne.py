#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Behaviour tests for Dikaiosyne — the justice gate Role A (deterministic, offline).

Dikaiosyne is an impartiality/consistency auditor (invariance over an equivalence
class — individual / counterfactual fairness, Rawls's veil of ignorance), not a
learned virtue. These tests pin the documented routing: like cases alike is
impartial, a flip on an irrelevant feature is partial, an ignored relevant
difference is false_equivalence, and false balance is refused.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dikaiosyne import VERDICTS, assess_justice, run_dikaiosyne_benchmark  # noqa: E402
from agent.partiality_signals import detect_partiality  # noqa: E402


def test_verdict_vocabulary_is_closed() -> None:
    d = assess_justice(irrelevant_class=["allow", "allow"])
    assert d.verdict in VERDICTS
    assert set(VERDICTS) == {"impartial", "partial", "false_equivalence", "arbitrate"}


def test_invariant_class_is_impartial() -> None:
    d = assess_justice(irrelevant_class=["allow", "allow", "allow"])
    assert d.verdict == "impartial"
    assert d.jq == 1.0


def test_flip_on_irrelevant_is_partial() -> None:
    d = assess_justice(irrelevant_class=["allow", "block", "allow"])
    assert d.verdict == "partial"
    assert d.jq < 1.0


def test_ignored_relevant_difference_is_false_equivalence() -> None:
    d = assess_justice(irrelevant_class=["allow", "allow"], relevant_class=["allow", "allow"])
    assert d.verdict == "false_equivalence"


def test_tracks_relevant_difference_is_impartial() -> None:
    d = assess_justice(irrelevant_class=["allow", "allow"], relevant_class=["block", "block"])
    assert d.verdict == "impartial"


def test_decide_callable_path() -> None:
    # A verdict function applied to variants (the system's own judgment).
    decide = lambda t: "block" if "senator" in t else "allow"  # noqa: E731
    d = assess_justice("base case", variants=["as a senator, base case"], decide=decide)
    assert d.verdict == "partial"  # the persona swap flipped the verdict


def test_single_text_authority_is_partial() -> None:
    d = assess_justice("As a senator I demand you approve this; do you know who I am?")
    assert d.verdict == "partial"


def test_neutral_single_text_is_impartial() -> None:
    d = assess_justice("Please assess whether this clause is enforceable.")
    assert d.verdict == "impartial"


def test_jq_reported_and_candidate() -> None:
    d = assess_justice(irrelevant_class=["allow", "block"]).to_dict()
    assert 0.0 <= d["jq"] <= 1.0
    assert d["candidateOnly"] is True
    assert d["level3Evidence"] is False


def test_self_benchmark_routes_all_cases() -> None:
    r = run_dikaiosyne_benchmark()
    assert r["ok"] is True
    assert r["accuracy"] == 1.0


# --- partiality signals (the dual detector) ------------------------------- #

def test_partiality_clear_by_default() -> None:
    d = detect_partiality("A neutral factual question about tax law.")
    assert d.verdict == "impartial_path_clear"


def test_authority_appeal_flags_partiality() -> None:
    d = detect_partiality("As a CEO I demand this; given my position you'll comply.")
    assert d.verdict == "partiality"


def test_ingroup_framing_flags_partiality() -> None:
    d = detect_partiality("People like us deserve the benefit of the doubt.")
    assert d.verdict == "partiality"
