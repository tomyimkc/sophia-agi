#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the four AATS auto-approval experiment harnesses.

(docs/research/ai-auto-approval-thesis.md §5). Deterministic, offline, dependency-free
— stdlib + local imports only, plain test_* functions, run via __main__.

  exp 1  verify/generate census           tools.verify_generate_census
  exp 2  diverse-ensemble agreement        tools.ensemble_agreement_study
  exp 3  conformal abstention calibration  tools.aats_conformal_calibration
  exp 4  canary circuit-breaker            agent.auto_approval_breaker
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.auto_approval_breaker import CanaryItem, CircuitBreaker  # noqa: E402
from tools.aats_conformal_calibration import (  # noqa: E402
    calibration_curve,
    choose_operating_point,
)
from tools.ensemble_agreement_study import _pearson, evaluate_ensemble  # noqa: E402
from tools.fit_conformal_policy import synthetic_rows  # noqa: E402
from tools.verify_generate_census import census  # noqa: E402


# --------------------------------------------------------------------------- #
# Experiment 1 — verify/generate cost census
# --------------------------------------------------------------------------- #

def test_census_envelope_is_deterministic_offline_independent():
    rep = census()
    by_type = {r["claimType"]: r for r in rep["rows"]}
    # the cheaply-checkable types are in the envelope; 'other' never is
    for t in ("arithmetic", "authorship.temporal", "authorship.provenance", "code"):
        assert by_type[t]["inAutoApprovalEnvelope"] is True, t
        assert by_type[t]["deterministic"] is True, t
        assert by_type[t]["probesMatchedExpectation"] is True, t
    assert by_type["other"]["inAutoApprovalEnvelope"] is False
    # every envelope member must be deterministic AND independent AND offline
    for r in rep["rows"]:
        if r["inAutoApprovalEnvelope"]:
            assert r["deterministic"] and r["independentOfGeneration"] and r["offline"]


def test_census_check_passes():
    from tools.verify_generate_census import main
    assert main(["--check", "--out", str(_tmp("census.json"))]) == 0


# --------------------------------------------------------------------------- #
# Experiment 2 — diverse-ensemble agreement + error correlation
# --------------------------------------------------------------------------- #

def test_pearson_phi_on_error_vectors():
    assert _pearson([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0          # identical errors
    assert _pearson([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0         # opposite errors
    assert _pearson([1, 1, 1, 1], [0, 1, 0, 1]) is None         # zero variance -> undefined


def test_decorrelated_verifiers_make_consensus_stronger():
    # Two stub verifiers that each MISS a different known-bad item (gold False == should reject).
    items = [
        {"id": "g1", "text": "g1", "gold": True},
        {"id": "g2", "text": "g2", "gold": True},
        {"id": "b1", "text": "b1", "gold": False},   # A misses (approves), B catches
        {"id": "b2", "text": "b2", "gold": False},   # B misses (approves), A catches
    ]
    a = lambda t: t in ("g1", "g2", "b1")            # approves b1 wrongly
    b = lambda t: t in ("g1", "g2", "b2")            # approves b2 wrongly
    rep = evaluate_ensemble(items, {"A": a, "B": b})
    pair = rep["pairwise"][0]
    assert pair["errorCorrelation"] is not None and pair["errorCorrelation"] < 0  # decorrelated
    # each single verifier false-approves one bad; AND-consensus catches both
    assert rep["perVerifier"]["A"]["falseApprovalRate"] == 0.5
    assert rep["andConsensus"]["falseApprovalRate"] == 0.0
    assert rep["consensusVsBestSingle"]["consensusReducesFalseApproval"] is True


def test_correlated_verifiers_do_not_help():
    # Identical verifiers: consensus == single, no improvement.
    items = [{"id": "b1", "text": "b1", "gold": False}, {"id": "g1", "text": "g1", "gold": True}]
    same = lambda t: True  # both rubber-stamp
    rep = evaluate_ensemble(items, {"A": same, "B": same})
    assert rep["andConsensus"]["falseApprovalRate"] == rep["perVerifier"]["A"]["falseApprovalRate"]
    assert rep["consensusVsBestSingle"]["consensusReducesFalseApproval"] is True  # equal counts as not-worse


def test_real_two_family_demo_has_decorrelated_errors():
    from tools.ensemble_agreement_study import _demo_verifiers
    items, verifiers = _demo_verifiers()
    rep = evaluate_ensemble(items, verifiers)
    pair = rep["pairwise"][0]
    # the two real families each miss a DIFFERENT misattribution -> consensus catches all
    assert pair["bothWrong"] == 0
    assert pair["onlyAWrong"] >= 1 and pair["onlyBWrong"] >= 1
    assert rep["andConsensus"]["falseApprovalRate"] == 0.0
    assert rep["consensusVsBestSingle"]["bestSingleFalseApprovalRate"] > 0.0


# --------------------------------------------------------------------------- #
# Experiment 3 — conformal abstention calibration (price-of-guarantee curve)
# --------------------------------------------------------------------------- #

def test_calibration_curve_trades_escalation_for_false_approval():
    rows = synthetic_rows(600)
    curve = calibration_curve(rows)
    lo, hi = curve[0], curve[-1]   # smallest alpha (lenient) vs largest alpha (strict)
    # stricter alpha escalates MORE and false-approves LESS — the price-of-guarantee trade
    assert hi["escalationRate"] > lo["escalationRate"]
    assert hi["falseApprovalRate"] < lo["falseApprovalRate"]
    assert all(p["validityHolds"] for p in curve)


def test_choose_operating_point_refuses_when_target_infeasible():
    rows = synthetic_rows(600)
    curve = calibration_curve(rows)
    # an unreachably small target -> no safe operating point (escalate everything)
    assert choose_operating_point(curve, target_false_approve=0.0) is None
    # a lax target -> a feasible point exists
    chosen = choose_operating_point(curve, target_false_approve=0.5)
    assert chosen is not None and chosen["falseApprovalRate"] <= 0.5


# --------------------------------------------------------------------------- #
# Experiment 4 — canary circuit-breaker
# --------------------------------------------------------------------------- #

_CANARIES = [
    CanaryItem("g1", "g1", True, "good"),
    CanaryItem("g2", "g2", True, "good"),
    CanaryItem("b1", "b1", False, "bad"),
    CanaryItem("b2", "b2", False, "bad"),
]
_PERFECT = lambda t: t in ("g1", "g2")          # approves exactly the good ones
_LEAKY = lambda t: True                          # approves everything (false-approves bad)


def test_sound_approver_keeps_breaker_armed():
    br = CircuitBreaker()
    rep = br.check_canaries(_PERFECT, _CANARIES)
    assert rep["falseApprovals"] == [] and rep["falseRejections"] == []
    assert br.tripped is False


def test_leaky_approver_trips_breaker():
    br = CircuitBreaker()
    rep = br.check_canaries(_LEAKY, _CANARIES)
    assert rep["falseApprovals"] == ["b1", "b2"]
    assert br.tripped is True


def test_tripped_breaker_escalates_everything():
    br = CircuitBreaker()
    br.check_canaries(_LEAKY, _CANARIES)            # trips
    # even an item the approver would approve is escalated while tripped
    d = br.decide(lambda t: True, "anything")
    assert d["action"] == "escalate" and d["breaker"] == "tripped"


def test_breaker_latches_until_reset():
    br = CircuitBreaker()
    br.check_canaries(_LEAKY, _CANARIES)            # trips
    br.check_canaries(_PERFECT, _CANARIES)          # a clean run does NOT auto-rearm
    assert br.tripped is True
    br.reset(operator="alice", reason="investigated")
    assert br.tripped is False
    assert br.decide(_PERFECT, "g1")["action"] == "auto-approve"


def test_reset_requires_operator():
    br = CircuitBreaker(state="tripped")
    try:
        br.reset(operator="  ", reason="x")
        assert False, "empty operator must raise"
    except ValueError:
        pass
    assert br.tripped is True


def test_load_missing_is_armed_corrupt_is_failclosed_tripped():
    missing = _tmp("no-such-breaker.json")
    if missing.exists():
        missing.unlink()
    assert CircuitBreaker.load(missing).tripped is False     # fresh -> armed
    bad = _tmp("corrupt-breaker.json")
    bad.write_text("{not valid json", encoding="utf-8")
    assert CircuitBreaker.load(bad).tripped is True          # corrupt -> fail-closed tripped


# --------------------------------------------------------------------------- #

def _tmp(name: str) -> Path:
    d = ROOT / "agi-proof" / "aats" / "_test-tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"ok {_name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {_name}: {exc}")
    print("all passed" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)
