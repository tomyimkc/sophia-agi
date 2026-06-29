# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for agent/step_verifier.py and agent/verified_reasoning_graph.py.

Physics cases run unconditionally (pure-Python SI units); math/sympy cases skip
when sympy is absent (the CI default fails closed to abstain, which is also tested).
"""
from __future__ import annotations

import pytest

from agent import math_verifier as mv
from agent import verified_reasoning_graph as vrg
from agent.step_verifier import Step, verify_derivation


def _has_sympy() -> bool:
    return mv.sympy_available()


# --- physics (always available) ------------------------------------------- #

def test_physics_clean_chain_accepted() -> None:
    res = verify_derivation(
        [{"expr": "30 N"}, {"expr": "30 kg*m/s^2"}], gold="30 N", default_domain="physics",
    )
    assert res.verdict == "accepted"
    assert res.vsc == 1.0


def test_physics_dimension_error_rejected() -> None:
    res = verify_derivation(
        [{"expr": "30 N"}, {"expr": "30 J"}], gold="30 N", default_domain="physics",
    )
    assert res.verdict == "rejected"


def test_physics_value_error_rejected() -> None:
    res = verify_derivation(
        [{"expr": "1 km"}, {"expr": "100 m"}], default_domain="physics",
    )
    assert res.verdict == "rejected"


# --- math (needs sympy) --------------------------------------------------- #

def test_math_clean_chain_accepted() -> None:
    if not _has_sympy():
        pytest.skip("sympy not installed")
    res = verify_derivation(
        [Step("(x+1)**2"), Step("x**2 + x + x + 1"), Step("x**2 + 2*x + 1")],
        gold="x**2 + 2*x + 1",
    )
    assert res.verdict == "accepted"
    assert res.n_accepted == res.n_transitions + 1  # transitions + gold check


def test_math_sign_misstep_rejected() -> None:
    if not _has_sympy():
        pytest.skip("sympy not installed")
    res = verify_derivation(
        [Step("(x+1)**2"), Step("x**2 - 2*x + 1")], gold="x**2 + 2*x + 1",
    )
    assert res.verdict == "rejected"


def test_math_abstains_without_sympy() -> None:
    if _has_sympy():
        pytest.skip("sympy present; this asserts the fail-closed path")
    res = verify_derivation([Step("(x+1)**2"), Step("x**2 + 2*x + 1")])
    assert res.verdict == "abstain"  # never a silent pass when uncheckable


# --- fail-closed aggregation --------------------------------------------- #

def test_single_step_no_gold_abstains() -> None:
    # Nothing to verify (no transition, no gold) -> never claim accepted.
    res = verify_derivation([{"expr": "5 W"}], default_domain="physics")
    assert res.verdict == "abstain"
    assert res.vsc == 0.0


def test_rejected_beats_abstain() -> None:
    # A real misstep dominates even if other steps are uncheckable.
    res = verify_derivation(
        [{"expr": "30 N"}, {"expr": "30 J"}], default_domain="physics",
    )
    assert res.verdict == "rejected"


# --- verified reasoning graph + certificate ------------------------------ #

def test_vrg_certificate_is_stable_and_tamper_evident() -> None:
    steps = [{"expr": "5 W"}, {"expr": "5 J/s"}]
    g1 = vrg.build_graph("watt to base", steps, gold="5 W", default_domain="physics")
    g2 = vrg.build_graph("watt to base", steps, gold="5 W", default_domain="physics")
    assert g1.certificate == g2.certificate and len(g1.certificate) == 64
    g3 = vrg.build_graph("watt to base", [{"expr": "5 W"}, {"expr": "5 J"}],
                         gold="5 W", default_domain="physics")
    assert g3.certificate != g1.certificate  # different content -> different hash
    assert g3.verdict == "rejected"


def test_solve_uses_proposer_but_verdict_is_machine_decided() -> None:
    # A confident-but-wrong proposer is still rejected by the oracle.
    def bad_proposer(_problem: str):
        return [{"expr": "30 N"}, {"expr": "30 J"}]

    g = vrg.solve("force in base SI", bad_proposer, gold="30 N", default_domain="physics")
    assert g.verdict == "rejected"
    assert g.to_dict()["canClaimAGI"] is False
