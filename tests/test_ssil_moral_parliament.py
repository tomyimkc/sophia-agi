#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the moral-parliament value gate (SSIL G1P). Offline, no torch.

Asserts the parliament's anti-dominance invariant (any hard veto rejects, even
against a large utilitarian sum), the moral-uncertainty approve floor, the quorum
fail-closed, the missing-input fail-closed, and the standard decision-dict shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_moral_parliament import (  # noqa: E402
    FRAMEWORKS,
    GATE_ID,
    SCHEMA,
    demo_bundle,
    evaluate,
)

_ALLOWED = {"promote", "quarantine", "reject"}


def _has(reasons: list[str], needle: str) -> bool:
    return any(needle in r for r in reasons)


# --- promote path ---------------------------------------------------------- #


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert d["schema"] == SCHEMA
    assert d["gate"] == GATE_ID
    assert _has(d["reasons"], "parliament quorum met")


def test_none_tuning_params_use_defaults() -> None:
    """Explicit None for the optional tuning params must NOT crash via
    float(None)/int(None); they coalesce to the defaults and still promote."""
    b = demo_bundle()
    b["vetoThreshold"] = None
    b["approveFloor"] = None
    b["minFrameworks"] = None
    d = evaluate(b)
    assert d["verdict"] == "promote", d["reasons"]
    # Defaults are applied: veto floor 0.8, approve floor 0.0, quorum 3.
    assert d["metrics"]["vetoThreshold"] == 0.8, d["metrics"]
    assert d["metrics"]["approveFloor"] == 0.0, d["metrics"]
    assert d["metrics"]["minFrameworks"] == 3, d["metrics"]


# --- reject reasons: one no-dominance veto per veto-capable lens ------------ #


def test_deontological_veto_rejects() -> None:
    """A large duty violation cannot be outvoted by a big utilitarian benefit."""
    d = evaluate(
        {
            "action": {"harm": 0.1, "dutyViolation": 0.9, "dishonesty": 0.0,
                       "fairness": 0.95, "careHarm": 0.0, "benefit": 1.0},
        }
    )
    assert d["verdict"] == "reject", d["reasons"]
    assert _has(d["reasons"], "framework deontological vetoes: no-dominance")


def test_contractualist_veto_rejects() -> None:
    """Gross unfairness is reasonably rejectable -> veto, despite high benefit."""
    d = evaluate(
        {
            "action": {"harm": 0.0, "dutyViolation": 0.0, "dishonesty": 0.0,
                       "fairness": 0.1, "careHarm": 0.0, "benefit": 1.0},
        }
    )
    assert d["verdict"] == "reject", d["reasons"]
    assert _has(d["reasons"], "framework contractualist vetoes: no-dominance")


def test_care_veto_rejects() -> None:
    """A severe breach of a dependent-care relationship vetoes."""
    d = evaluate(
        {
            "action": {"harm": 0.0, "dutyViolation": 0.0, "dishonesty": 0.0,
                       "fairness": 0.95, "careHarm": 0.9, "benefit": 1.0},
        }
    )
    assert d["verdict"] == "reject", d["reasons"]
    assert _has(d["reasons"], "framework care vetoes: no-dominance")


def test_veto_not_outvoted_by_weight() -> None:
    """No-dominance: even with the vetoing lens given tiny weight and the
    consequentialist given huge weight, the veto still forces reject."""
    d = evaluate(
        {
            "action": {"harm": 0.0, "dutyViolation": 0.9, "dishonesty": 0.0,
                       "fairness": 0.95, "careHarm": 0.0, "benefit": 1.0},
            "weights": {"consequentialist": 100.0, "deontological": 0.01,
                        "virtue": 1.0, "contractualist": 1.0, "care": 1.0},
        }
    )
    assert d["verdict"] == "reject", d["reasons"]
    assert _has(d["reasons"], "no-dominance")


# --- quarantine / abstain reasons ------------------------------------------ #


def test_below_approve_floor_quarantines() -> None:
    """No veto, but the weighted-mean approval is below the floor -> abstain."""
    d = evaluate(
        {
            "action": {"harm": 0.3, "dutyViolation": 0.3, "dishonesty": 0.3,
                       "fairness": 0.5, "careHarm": 0.2, "benefit": 0.3},
            "approveFloor": 0.6,
        }
    )
    assert d["verdict"] == "quarantine", d["reasons"]
    assert _has(d["reasons"], "below approve floor")
    assert _has(d["reasons"], "abstained")


def test_too_few_frameworks_quarantines() -> None:
    """Fewer present frameworks than the quorum -> fail-closed quarantine."""
    d = evaluate(
        {
            "action": demo_bundle()["action"],
            "weights": {"consequentialist": 1.0, "virtue": 1.0},
            "minFrameworks": 3,
        }
    )
    assert d["verdict"] == "quarantine", d["reasons"]
    assert _has(d["reasons"], "too few frameworks present")


# --- fail-closed: missing required inputs ---------------------------------- #


def test_missing_bundle_quarantines() -> None:
    d = evaluate(None)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert _has(d["reasons"], "missing required input: bundle")


def test_missing_action_quarantines() -> None:
    d = evaluate({"weights": {"consequentialist": 1.0}})
    assert d["verdict"] == "quarantine", d["reasons"]
    assert _has(d["reasons"], "missing required input: action")


def test_missing_action_feature_quarantines() -> None:
    """A missing safety feature must NOT default to a passing value -> quarantine."""
    a = dict(demo_bundle()["action"])
    del a["benefit"]
    d = evaluate({"action": a})
    assert d["verdict"] == "quarantine", d["reasons"]
    assert _has(d["reasons"], "missing required input: action.benefit")


# --- standard decision-dict invariants ------------------------------------- #


def test_decision_dict_invariants() -> None:
    for bundle in (demo_bundle(), {"action": demo_bundle()["action"], "approveFloor": 0.99}, None):
        d = evaluate(bundle)
        assert d["canClaimAGI"] is False
        assert d["candidateOnly"] is True
        assert d["level3Evidence"] is False
        assert d["verdict"] in _ALLOWED
        assert isinstance(d["reasons"], list) and d["reasons"]
        assert isinstance(d["boundary"], str) and d["boundary"]
        assert d["candidateId"] == "sophia-rlvr-v1"


def test_framework_set_is_the_five_lenses() -> None:
    assert set(FRAMEWORKS) == {
        "consequentialist", "deontological", "virtue", "contractualist", "care",
    }
    d = evaluate(demo_bundle())
    assert set(d["metrics"]["frameworks"]) == set(FRAMEWORKS)


def main() -> int:
    test_demo_bundle_promotes()
    test_none_tuning_params_use_defaults()
    test_deontological_veto_rejects()
    test_contractualist_veto_rejects()
    test_care_veto_rejects()
    test_veto_not_outvoted_by_weight()
    test_below_approve_floor_quarantines()
    test_too_few_frameworks_quarantines()
    test_missing_bundle_quarantines()
    test_missing_action_quarantines()
    test_missing_action_feature_quarantines()
    test_decision_dict_invariants()
    test_framework_set_is_the_five_lenses()
    print("test_ssil_moral_parliament: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
