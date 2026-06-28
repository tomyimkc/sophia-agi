#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Datalog port of ``provenance_faithful``.

Covers two layers:
  - the Datalog engine itself (stratified negation, fixpoint, query) — pure logic,
    no provenance records needed.
  - the faithfulness of ``check_claim_datalog`` vs ``check_claim`` on the
    hand-built carve-out cases that are the gate's trickiest behaviors.

The full 319-case byte-identical audit lives in ``tools/run_datalog_provenance_audit.py``
(it is slow because it exercises every committed case × 3 variants).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.datalog_engine import A, Program, Rule, Var  # noqa: E402
from agent.guarded import check_claim  # noqa: E402
from agent.datalog_provenance import check_claim_datalog  # noqa: E402


# --------------------------------------------------------------------------- #
# Engine: pure-logic correctness
# --------------------------------------------------------------------------- #
def test_engine_positive_rule_derives_fact():
    p = Program()
    p.fact(A("parent", "a", "b"))
    p.rule(Rule.clause(A("ancestor", Var("X"), Var("Y")), A("parent", Var("X"), Var("Y"))))
    assert ("a", "b") in p.query(A("ancestor", Var("X"), Var("Y")))


def test_engine_recursive_transitive_closure():
    p = Program()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d")]:
        p.fact(A("edge", x, y))
    p.rule(Rule.clause(A("reach", Var("X"), Var("Y")), A("edge", Var("X"), Var("Y"))))
    p.rule(Rule.clause(A("reach", Var("X"), Var("Z")), A("edge", Var("X"), Var("Y")), A("reach", Var("Y"), Var("Z"))))
    reached = sorted(y for (x, y) in p.query(A("reach", "a", Var("Y"))))
    assert reached == ["b", "c", "d"]


def test_engine_stratified_negation_as_failure():
    # The abstention rule shape: violation :- asserted(C,W,A), forbidden(W,A), not carveout(C).
    p = Program()
    p.fact(A("forbidden", "W", "Bad"))
    p.fact(A("asserted_in_clause", "c1", "W", "Bad"))   # c1 asserts, NO carveout -> violation
    p.fact(A("asserted_in_clause", "c2", "W", "Bad"))   # c2 asserts, HAS carveout -> no violation
    p.fact(A("carveout", "c2"))
    p.rule(
        Rule.clause(
            A("violation", Var("W"), Var("A")),
            A("asserted_in_clause", Var("C"), Var("W"), Var("A")),
            A("forbidden", Var("W"), Var("A")),
            A("carveout", Var("C")).neg(),
        )
    )
    viol = sorted(a.args for a in p.solve() if a.pred == "violation")
    assert viol == [("W", "Bad")]


def test_engine_rejects_negation_cycle_as_unstratifiable():
    p = Program()
    p.rule(Rule.clause(A("p"), A("q").neg()))
    p.rule(Rule.clause(A("q"), A("p").neg()))
    with pytest.raises(ValueError, match="not stratifiable"):
        p.solve()


def test_engine_negation_reads_earlier_stratum_only():
    # carveout(c) blocks violation even when both rules are in the program.
    p = Program()
    p.fact(A("forbidden", "W", "Bad"))
    p.fact(A("asserted_in_clause", "c", "W", "Bad"))
    p.fact(A("carveout", "c"))
    p.rule(
        Rule.clause(
            A("violation", Var("W"), Var("A")),
            A("asserted_in_clause", Var("C"), Var("W"), Var("A")),
            A("forbidden", Var("W"), Var("A")),
            A("carveout", Var("C")).neg(),
        )
    )
    assert p.query(A("violation", Var("W"), Var("A"))) == set()


# --------------------------------------------------------------------------- #
# Faithfulness: Datalog port == Python gate on the tricky carve-out cases
# --------------------------------------------------------------------------- #
FAITHFUL_CASES = [
    ("affirm forbidden", "Confucius wrote the Dao De Jing."),
    ("myth carveout", "It is a myth that Confucius wrote the Dao De Jing."),
    ("negation carveout", "Confucius did not write the Dao De Jing; Laozi did."),
    ("misconception carveout", "It is a misconception that Confucius wrote the Dao De Jing."),
    ("contrast clause split", "Confucius wrote the Dao De Jing but the attribution is disputed."),
    ("unrelated text", "Completely unrelated text about cooking pasta."),
    ("gold author pass", "Laozi wrote the Dao De Jing."),
]


@pytest.mark.parametrize("name,text", FAITHFUL_CASES)
def test_datalog_matches_python_gate(name, text):
    """The Datalog-derived verdict must equal the Python gate's verdict,
    including the violations list, on every carve-out behavior."""
    py = check_claim(text)
    dl = check_claim_datalog(text)
    assert py["passed"] == dl["passed"], f"{name}: passed differs py={py['passed']} dl={dl['passed']}"
    assert py["violations"] == dl["violations"], (
        f"{name}: violations differ py={py['violations']} dl={dl['violations']}"
    )


def test_datalog_backend_fails_closed_on_asserted_forbidden():
    """The whole point: an asserted forbidden attribution must FAIL under Datalog,
    not silently pass because the fact extraction missed it."""
    dl = check_claim_datalog("Confucius wrote the Dao De Jing.")
    assert dl["passed"] is False
    assert dl["violations"] == ["confucius -> dao_de_jing"]


def test_check_claim_datalog_has_same_shape_as_check_claim():
    """Public contract: same keys, same types, boolean passed, list violations."""
    dl = check_claim_datalog("innocuous text")
    assert set(dl.keys()) == {"passed", "reasons", "violations"}
    assert isinstance(dl["passed"], bool)
    assert isinstance(dl["violations"], list)
    assert isinstance(dl["reasons"], list)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
