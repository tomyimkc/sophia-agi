#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ko-detector tests — GO ko-rule on an iterative revision sequence.

A ko = a belief state (abstain set) that recurs within the KO_MAX_ROUNDS window.
The detector's contract: a ko -> escalate (NEVER silently abstain), because a ko
is an irreducible oscillation, not a terminal "do not assert".

Dependency-free, offline, deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.consequence import KO_MAX_ROUNDS, detect_ko, is_ko  # noqa: E402


def test_no_recurrence_is_not_ko() -> None:
    seq = [{"a"}, {"a", "b"}, {"a", "b", "c"}]  # strictly growing -> no recurrence
    alert = detect_ko(seq)
    assert alert.ko is False
    assert alert.recommendedVerdict == "allow"
    assert alert.rounds == 3


def test_immediate_recurrence_is_ko() -> None:
    # oscillation: abstain X, reassert X, abstain X -> the {} <-> {X} ping-pong
    seq = [{"x"}, set(), {"x"}]
    alert = detect_ko(seq)
    assert alert.ko is True
    assert alert.cycle[0] == 0 and alert.cycle[1] == 2
    assert alert.recommendedVerdict == "escalate"
    assert "ko" in alert.reason


def test_recurrence_outside_window_is_not_ko() -> None:
    # {x} recurs at distance KO_MAX_ROUNDS+1 (4 intermediate distinct states ->
    # indices 0..5, distance 5 > 4) -> not a ko.
    n_intermediate = KO_MAX_ROUNDS  # distance = n_intermediate + 1 = KO_MAX_ROUNDS+1
    seq = [{"x"}] + [{f"s{i}"} for i in range(n_intermediate)] + [{"x"}]
    alert = detect_ko(seq)
    assert alert.ko is False


def test_recurrence_just_inside_window_is_ko() -> None:
    # {x} recurs at distance exactly KO_MAX_ROUNDS (3 intermediate distinct states
    # -> indices 0..4, distance 4 <= 4) -> ko at the boundary (<= is inclusive).
    n_intermediate = KO_MAX_ROUNDS - 1  # distance = n_intermediate + 1 = KO_MAX_ROUNDS
    seq = [{"x"}] + [{f"s{i}"} for i in range(n_intermediate)] + [{"x"}]
    alert = detect_ko(seq)
    assert alert.ko is True


def test_ko_uses_set_equality_not_text() -> None:
    # two rounds with the SAME abstain set but different "reasoning" are ko-equal
    seq = [{"a", "b"}, {"b", "a"}]  # order-independent
    alert = detect_ko(seq)
    assert alert.ko is True


def test_empty_and_single_round_sequences_are_safe() -> None:
    assert detect_ko([]).ko is False
    assert detect_ko([{"x"}]).ko is False
    assert is_ko([]) is False


def test_ko_recommendation_is_escalate_not_abstain() -> None:
    # the load-bearing invariant: ko must escalate, never silently abstain
    seq = [{"x"}, set(), {"x"}]
    alert = detect_ko(seq)
    assert alert.ko is True
    assert alert.recommendedVerdict == "escalate"
    assert alert.recommendedVerdict != "abstain"


def test_to_dict_shape() -> None:
    alert = detect_ko([{"x"}, set(), {"x"}])
    d = alert.to_dict()
    assert d["schema"] == "sophia.consequence.ko.v1"
    assert d["ko"] is True
    assert d["candidateOnly"] is True and d["level3Evidence"] is False
    assert "AGI proof" in d["boundary"]


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_ko_detector: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
