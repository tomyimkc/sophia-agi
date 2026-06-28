#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the self-extending verification flywheel (selfextend/). Deterministic,
offline. Each falsifiable claim from the path-to-AGI brainstorm has a check here."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfextend import (  # noqa: E402
    AbstentionLedger, CausalGraph, CompetenceMap, brier_score,
    expected_calibration_error, propose_and_validate, reward_is_hackable, run_flywheel,
    run_long_horizon, run_transfer, verified_reward, verify_by_execution,
)

# domains the loop has never seen: a different signal token per domain, stable across
# the train/held-out split (so a synthesized verifier can be validated to generalize)
_DOMAINS = {
    "danger": [("delete the database", True), ("delete user files", True),
               ("please delete records", True), ("delete everything now", True),
               ("read the database", False), ("read user files", False),
               ("please read records", False), ("read everything now", False)],
    "question": [("what is this", True), ("what happened here", True),
                 ("what time is it", True), ("what do you mean", True),
                 ("this is fine", False), ("it happened here", False),
                 ("the time is noon", False), ("you mean well", False)],
}


# --------------------------------------------------- abstention ledger / curiosity
def test_abstention_ledger_agenda() -> None:
    led = AbstentionLedger()
    for _ in range(3):
        led.record(domain="law")
    led.record(domain="finance")
    assert led.agenda(1)[0] == ("law", 3)
    assert "law" in led.gap_domains()


# --------------------------------------------------- verifier synthesis (validate)
def test_synthesis_promotes_only_when_validated() -> None:
    from selfextend.verifier_synthesis import stratified_split
    train, heldout = stratified_split(_DOMAINS["danger"])
    out = propose_and_validate(train, heldout, threshold=0.8)
    assert out["promoted"] is True and out["heldoutAccuracy"] >= 0.8


def test_synthesis_abstains_on_unlearnable() -> None:
    # random labels -> no rule should validate -> stay abstained (fail-closed)
    noise = [("alpha", True), ("beta", False), ("gamma", True), ("delta", False),
             ("alpha", False), ("beta", True), ("gamma", False), ("delta", True)]
    out = propose_and_validate(noise[:4], noise[4:], threshold=0.8)
    assert out["promoted"] is False


# --------------------------------------------------- keystone flywheel
def test_flywheel_raises_coverage_without_gaming() -> None:
    r = run_flywheel(_DOMAINS, threshold=0.8)
    assert r["coverageBefore"] == 0.0
    assert r["coverageAfter"] > 0.0                 # synthesized + validated new verifiers
    assert r["heldoutFalseAcceptRate"] <= 0.25      # promotion did not game the bar


# --------------------------------------------------- calibration metrics
def test_calibration_metrics() -> None:
    perfect = [(1.0, True), (0.0, False), (1.0, True), (0.0, False)]
    assert expected_calibration_error(perfect) == 0.0 and brier_score(perfect) == 0.0
    bad = [(1.0, False), (1.0, False)]
    assert brier_score(bad) == 1.0 and expected_calibration_error(bad) > 0.5


# --------------------------------------------------- competence map (self-model)
def test_competence_routing() -> None:
    cm = CompetenceMap(threshold=0.7)
    for _ in range(10):
        cm.update("history", True)
    for _ in range(10):
        cm.update("speculation", False)
    assert cm.route("history") == "answer"
    assert cm.route("speculation") == "abstain"


# --------------------------------------------------- causal world model
def test_causal_beats_correlation_under_confounding() -> None:
    # confounder C -> X, C -> Y, and a true X -> Y edge. Observational coef is biased;
    # do-effect recovers the direct causal weight.
    g = CausalGraph().add_edge("C", "X", 2.0).add_edge("C", "Y", 3.0).add_edge("X", "Y", 1.0)
    assert g.causal_effect("X", "Y") == 1.0          # direct path only
    assert g.confounded("X", "Y")                    # observation != causation
    assert g.observational_coef("X", "Y") != 1.0


def test_causal_no_effect_when_no_path() -> None:
    g = CausalGraph().add_edge("C", "X", 2.0).add_edge("C", "Y", 3.0)  # X does NOT cause Y
    assert g.causal_effect("X", "Y") == 0.0          # do(X) has no effect
    assert g.observational_coef("X", "Y") != 0.0     # but they ARE correlated (via C)


# --------------------------------------------------- cross-domain transfer
def test_transfer_across_builtin_domains() -> None:
    r = run_transfer(threshold=0.75)
    assert r["transferred"] is True and r["promotedCount"] == len(r["domains"])


# --------------------------------------------------- environment-as-verifier
def test_env_verifier_executes() -> None:
    assert verify_by_execution("arithmetic", "6*7", {"expected": 42})["passed"] is True
    assert verify_by_execution("arithmetic", "6*7", {"expected": 41})["passed"] is False
    assert verify_by_execution("arithmetic", "__import__('os')", {"expected": 0})["passed"] is False
    rx = verify_by_execution("regex", r"\d+", {"cases": [("abc 12", True), ("abc", False)]})
    assert rx["passed"] is True


# --------------------------------------------------- verified reward + anti-gaming
def test_verified_reward_and_hacking_detector() -> None:
    train_v = lambda c: "magicword" in c           # a hackable train verifier
    held_v = lambda c: c.strip().endswith(".")     # the real held-out check
    gamed = ["magicword", "magicword", "magicword"]    # games train, fails held-out
    assert verified_reward("magicword", train_v) == 1.0
    rep = reward_is_hackable(gamed, train_v, held_v)
    assert rep["hacked"] is True and rep["trainReward"] > rep["heldoutReward"]


# --------------------------------------------------- long-horizon with recovery
def test_long_horizon_recovers_then_drifts() -> None:
    steps = [
        {"id": "s1", "content": "step one", "sources": ["src1"]},                       # accept
        {"id": "s2", "content": "step two", "sources": [], "repair_sources": ["src2"]}, # held -> repair -> accept
        {"id": "s3", "content": "step three", "sources": [{"id": "bad", "status": "refuted"}]},  # rejected -> drift
        {"id": "s4", "content": "never reached", "sources": ["src4"]},
    ]
    r = run_long_horizon(steps, max_repairs=1)
    assert r["completedSteps"] == 2 and r["recoveries"] == 1 and r["driftedAt"] == "s3"
    assert r["effectiveHorizon"] == 2


# --------------------------------------------------- closed loop (the capstone)
def _divisibility_domain():
    """Non-trivial held-out domain: signal is divisibility-by-5 (numeric), NOT a single
    lexical token. The real agent.verifier_synthesis engine composes a `divisible_by_5`
    verifier and validates it on a disjoint split; a single-token stump provably cannot
    express this (it plateaus near chance)."""
    valid = [str(n) for n in range(5, 101, 5)]  # 20 multiples of 5
    invalid = ["7", "13", "22", "31", "44", "48", "52", "61", "77", "88",
               "99", "103", "ab", "3x", "12", "19"]  # none divisible by 5
    return [(v, True) for v in valid] + [(i, False) for i in invalid]


def test_loop_closes_on_heldout_domain() -> None:
    from selfextend import close_loop
    r = close_loop("divisible_by_5", _divisibility_domain())
    assert r["loop_closed"] is True
    assert r["promoted"] and r["postAccuracy"] > r["preAccuracy"]
    assert r["routeBefore"] == "abstain" and r["routeAfter"] == "answer"
    assert all(r["invariants"].values())
    # the real engine synthesized a compositional gate (not a single-token stump)
    assert r["rule"]["engine"] == "agent.verifier_synthesis"
    assert "divisible" in r["rule"]["gate"]


def test_loop_stays_abstained_when_unlearnable() -> None:
    from selfextend import close_loop
    noise = [(w, i % 2 == 0) for i, w in enumerate(
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lam", "mu"])]
    r = close_loop("noise", noise)
    assert r["loop_closed"] is False and r["promoted"] is False  # fail-closed


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_selfextend: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
