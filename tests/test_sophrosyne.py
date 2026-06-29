#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Behaviour tests for Sophrosyne — the temperance gate (deterministic, offline).

Sophrosyne is a measure/magnitude DECISION HEURISTIC (Aristotle's mean over an
expenditure-vs-demand deviation), not a learned virtue. These tests pin the
documented routing so the no-overclaim boundary stays honest: excess is restrained,
deficiency is sustained, akrasia escalates, and a required verification step is
never cut (temperance is not negligence).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.intemperance_signals import detect_intemperance  # noqa: E402
from agent.sophrosyne import VERDICTS, assess_temperance, run_sophrosyne_benchmark  # noqa: E402


def test_verdict_vocabulary_is_closed() -> None:
    d = assess_temperance("anything", context={"demand": 0.5, "expenditure": 0.5})
    assert d.verdict in VERDICTS
    # Sophrosyne keeps its OWN vocabulary (not a conscience verdict).
    assert set(VERDICTS) == {"proportionate", "restrain", "sustain", "escalate"}


def test_expenditure_tracking_demand_is_proportionate() -> None:
    d = assess_temperance("Answer directly.", context={"demand": 0.5, "expenditure": 0.5, "marginalValue": 0.5})
    assert d.verdict == "proportionate"
    assert abs(d.mq) <= 0.15


def test_efficient_short_answer_is_not_nagged() -> None:
    # Spending little on a cheap task with little left to gain is the mean, not deficiency.
    d = assess_temperance("Short answer.", context={"demand": 0.25, "expenditure": 0.3, "marginalValue": 0.3})
    assert d.verdict == "proportionate"


def test_excess_low_marginal_value_restrains() -> None:
    d = assess_temperance("Pad with detail nobody asked for.",
                          context={"demand": 0.3, "expenditure": 0.85, "marginalValue": 0.2})
    assert d.verdict == "restrain"
    assert d.mq > 0


def test_runaway_loop_restrains() -> None:
    d = assess_temperance("Keep iterating on the same point.",
                          context={"demand": 0.4, "expenditure": 0.9, "marginalValue": 0.25,
                                   "loopIterations": 5, "frontierShrinking": False})
    assert d.verdict == "restrain"
    assert d.intemperance["axis"] == "excess"


def test_deficiency_high_marginal_value_sustains() -> None:
    d = assess_temperance("Stop here.",
                          context={"demand": 0.7, "expenditure": 0.2, "marginalValue": 0.8,
                                   "budgetRemaining": 0.8, "proposedStop": True})
    assert d.verdict == "sustain"
    assert d.mq < 0


def test_akrasia_on_contested_budget_escalates() -> None:
    d = assess_temperance("I want to keep going and add more.",
                          context={"demand": 0.5, "expenditure": 0.6, "marginalValue": 0.5,
                                   "appetite": 0.8, "budgetRemaining": 0.2})
    assert d.verdict == "escalate"


def test_forces_are_reported_for_audit() -> None:
    d = assess_temperance("x", context={"demand": 0.4, "expenditure": 0.6}).to_dict()
    assert set(d["forces"]) == {"delta", "epsilon", "mu", "alpha", "rho"}
    assert d["candidateOnly"] is True
    assert d["level3Evidence"] is False


def test_self_benchmark_routes_all_cases() -> None:
    r = run_sophrosyne_benchmark()
    assert r["ok"] is True
    assert r["accuracy"] == 1.0
    assert r["candidateOnly"] is True


# --- intemperance signals (the dual detector) ----------------------------- #

def test_intemperance_clear_by_default() -> None:
    d = detect_intemperance("A normal proportionate sentence.")
    assert d.verdict == "measure_clear"
    assert d.axis == "none"


def test_hedge_stacking_flags_excess() -> None:
    d = detect_intemperance("I think perhaps it maybe could possibly be the case, arguably.",
                            context={"demand": 0.4, "expenditure": 0.8})
    assert d.axis == "excess"


def test_truncation_flags_deficiency() -> None:
    d = detect_intemperance("The rest is left as an exercise. TODO",
                            context={"demand": 0.75, "expenditure": 0.3})
    assert d.axis == "deficiency"


def test_signal_never_forces_an_action() -> None:
    # The dual module is informational: it returns a verdict/axis, no action verb.
    d = detect_intemperance("Pad pad pad pad pad pad pad pad.").to_dict()
    assert d["verdict"] in {"measure_clear", "excess_risk", "excess", "deficiency_risk", "deficiency"}
