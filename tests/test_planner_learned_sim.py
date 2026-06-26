#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the learned planner simulator (offline, deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import planner_learned_sim as pls  # noqa: E402
from agent import planner_mcts as pm  # noqa: E402


class _ScriptedPredictor:
    """Returns a fixed p regardless of input — lets tests force the learned
    outcome mapping deterministically."""

    def __init__(self, p: float):
        self.p = p

    def predict(self, state: str, action: str) -> float:
        return self.p


class _BrokenPredictor:
    """Always raises — exercises the fail-closed scripted fallback."""

    def predict(self, state, action):
        raise RuntimeError("model offline")


def test_high_confidence_predictor_drives_outcome() -> None:
    """A predictor confidently saying success (p=0.9) maps a source action to 'entails'."""
    sim = pls.LearnedSimulator(_ScriptedPredictor(0.9))
    state = pm.initial_state("Aristotle wrote the Nicomachean Ethics.")
    action = pm.Action("crossref_openalex", cost=0.6, source_gain=1)
    assert sim.outcome(state, action) == "entails"
    assert sim.stats()["predicted"] >= 1


def test_low_confidence_predictor_contradicts_judge() -> None:
    """A confident-low p (0.1) on a judge action => 'contradicts' (evidence against)."""
    sim = pls.LearnedSimulator(_ScriptedPredictor(0.1))
    state = pm.initial_state("A claim needing a judge.")
    judge = pm.Action("competent_judge_with_evidence", cost=1.0, judge=True)
    assert sim.outcome(state, judge) == "contradicts"


def test_uncertain_predictor_falls_back_to_scripted() -> None:
    """A predictor at chance (p~0.5) is uninformative => fall back to the scripted
    rule (fail-closed: a weak model must not steer the search)."""
    sim = pls.LearnedSimulator(_ScriptedPredictor(0.5), min_confidence=0.1)
    state = pm.initial_state("Aristotle wrote the Nicomachean Ethics.")
    action = pm.Action("crossref_openalex", cost=0.6, source_gain=1)
    out = sim.outcome(state, action)
    # scripted rule for a non-special source action => "entails"
    assert out == "entails"
    assert sim.stats()["fallbackToScripted"] >= 1
    assert sim.stats()["predicted"] == 0


def test_broken_predictor_falls_back_to_scripted() -> None:
    """A predictor that raises must not crash the planner — fall back to scripted."""
    sim = pls.LearnedSimulator(_BrokenPredictor())
    state = pm.initial_state("Some claim.")
    action = pm.Action("crossref_openalex", cost=0.6, source_gain=1)
    assert sim.outcome(state, action) == "entails"  # scripted default


def test_run_mcts_with_model_returns_valid_plan() -> None:
    """The planner produces a structured plan over the learned model, with the
    learned-simulator stats attached."""
    plan = pls.run_mcts_with_model(
        "Aristotle wrote the Nicomachean Ethics.",
        _ScriptedPredictor(0.85),
        iterations=40,
        seed=0,
    )
    assert plan["schema"] == "sophia.verification_mcts_plan.v1"
    assert plan["simulatorKind"] == "learned"
    assert "learnedSimulator" in plan
    assert isinstance(plan["plan"], list)
    # candidate discipline preserved
    assert plan["candidateOnly"] is True and plan["level3Evidence"] is False


def test_profiles_still_override_learned_model() -> None:
    """Live adapters / tests can force an outcome via profiles, winning over the
    learned model — the escape hatch stays intact."""
    sim = pls.LearnedSimulator(_ScriptedPredictor(0.9), profiles={"Aristotle": {"crossref_openalex": "none"}})
    state = pm.initial_state("Aristotle wrote X.")
    action = pm.Action("crossref_openalex", cost=0.6, source_gain=1)
    assert sim.outcome(state, action) == "none"  # profile wins


def main() -> int:
    test_high_confidence_predictor_drives_outcome()
    test_low_confidence_predictor_contradicts_judge()
    test_uncertain_predictor_falls_back_to_scripted()
    test_broken_predictor_falls_back_to_scripted()
    test_run_mcts_with_model_returns_valid_plan()
    test_profiles_still_override_learned_model()
    print("test_planner_learned_sim: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
