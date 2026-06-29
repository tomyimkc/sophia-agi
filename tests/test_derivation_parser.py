# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for agent/derivation_parser.py."""
from __future__ import annotations

from agent.derivation_parser import parse_derivation


def test_explicit_step_lines() -> None:
    text = "STEP: (x+1)**2 | start\nSTEP: x**2 + 2*x + 1 | expand"
    steps = parse_derivation(text)
    assert [s.expr for s in steps] == ["(x+1)**2", "x**2 + 2*x + 1"]
    assert steps[0].rule == "start"


def test_equality_chain() -> None:
    steps = parse_derivation("so (x+2)*(x-2) = x**2 - 2*x + 2*x - 4 = x**2 - 4")
    assert len(steps) == 3
    assert steps[-1].expr == "x**2 - 4"


def test_equality_chain_ignores_relational() -> None:
    # '>=' and '==' must not be treated as step separators.
    steps = parse_derivation("if x >= 0 then x == x")
    assert len(steps) < 2  # no spurious split


def test_answer_only_fallback() -> None:
    steps = parse_derivation("After working it out, the answer is 5/6.")
    assert len(steps) == 1
    assert "5/6" in steps[0].expr


def test_empty_text_is_no_steps() -> None:
    assert parse_derivation("") == []
    assert parse_derivation("I have no idea.") == []
