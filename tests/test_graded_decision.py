#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/graded_decision.py (offline, deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import graded_decision as gd  # noqa: E402
from agent.corroboration import Evidence  # noqa: E402


def test_pass_high_confidence_answers() -> None:
    d = gd.decide(gate_passed=True, confidence=0.9)
    assert d["action"] == "answer", d


def test_pass_low_confidence_is_suspicious() -> None:
    # a low-confidence pass should NOT be a clean answer
    d = gd.decide(gate_passed=True, confidence=0.2)
    assert d["action"] == "abstain", d
    # a mid pass hedges
    d_mid = gd.decide(gate_passed=True, confidence=0.5)
    assert d_mid["action"] == "hedge", d_mid


def test_fail_high_confidence_hedges() -> None:
    d = gd.decide(gate_passed=False, confidence=0.85)
    assert d["action"] == "hedge", d


def test_fail_low_confidence_abstains() -> None:
    d = gd.decide(gate_passed=False, confidence=0.3, violations=["forbidden attribution"])
    assert d["action"] == "abstain", d
    assert d["n_violations"] == 1


def test_thresholds_override() -> None:
    # raising hi turns a former answer into a hedge
    d = gd.decide(gate_passed=True, confidence=0.75, thresholds={"hi": 0.8})
    assert d["action"] == "hedge", d
    # bad thresholds rejected
    try:
        gd.decide(gate_passed=True, confidence=0.5, thresholds={"hi": 0.2, "lo": 0.5})
        assert False, "expected ValueError for lo > hi"
    except ValueError:
        pass


def test_confidence_bounds_validated() -> None:
    for bad in (-0.1, 1.5):
        try:
            gd.decide(gate_passed=True, confidence=bad)
            assert False, f"expected ValueError for confidence={bad}"
        except ValueError:
            pass


def test_confidence_neutral_default() -> None:
    assert gd.answer_confidence() == 0.5


def test_confidence_rises_with_agreeing_corroboration() -> None:
    one = gd.answer_confidence(corroboration_evidence=[Evidence("s0", 0.7, "g0")])
    two = gd.answer_confidence(
        corroboration_evidence=[Evidence("s0", 0.7, "g0"), Evidence("s1", 0.7, "g1")]
    )
    three = gd.answer_confidence(
        corroboration_evidence=[
            Evidence("s0", 0.7, "g0"),
            Evidence("s1", 0.7, "g1"),
            Evidence("s2", 0.7, "g2"),
        ]
    )
    assert one < two < three, (one, two, three)


def test_confidence_falls_with_dissent() -> None:
    agree = gd.answer_confidence(
        corroboration_evidence=[Evidence("a", 0.7, "g0"), Evidence("b", 0.7, "g1")]
    )
    with_dissent = gd.answer_confidence(
        corroboration_evidence=[
            Evidence("a", 0.7, "g0"),
            Evidence("b", 0.7, "g1"),
            Evidence("c", 0.2, "g2"),
        ]
    )
    assert with_dissent < agree, (agree, with_dissent)


def test_confidence_from_self_consistency() -> None:
    # 3/4 agreeing -> 0.75
    conf = gd.answer_confidence(self_consistency_samples=["x", "x", "x", "y"])
    assert abs(conf - 0.75) < 1e-9, conf
    # corroboration takes precedence when both are supplied
    both = gd.answer_confidence(
        corroboration_evidence=[Evidence("s0", 0.9, "g0")],
        self_consistency_samples=["x", "x"],
    )
    assert abs(both - 0.9) < 1e-6, both


def test_end_to_end_corroboration_drives_action() -> None:
    # strong independent agreement on a clean gate -> answer
    strong = [Evidence("s0", 0.8, "g0"), Evidence("s1", 0.8, "g1"), Evidence("s2", 0.8, "g2")]
    conf = gd.answer_confidence(corroboration_evidence=strong)
    assert gd.decide(gate_passed=True, confidence=conf)["action"] == "answer"


def main() -> int:
    test_pass_high_confidence_answers()
    test_pass_low_confidence_is_suspicious()
    test_fail_high_confidence_hedges()
    test_fail_low_confidence_abstains()
    test_thresholds_override()
    test_confidence_bounds_validated()
    test_confidence_neutral_default()
    test_confidence_rises_with_agreeing_corroboration()
    test_confidence_falls_with_dissent()
    test_confidence_from_self_consistency()
    test_end_to_end_corroboration_drives_action()
    print("test_graded_decision: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
