#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the conformal abstention gate (C1) and its decision wiring."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.conformal_gate import (
    ConformalPolicy,
    conformal_risk_control,
    crc_validity_check,
    evaluate_policy,
    fit_conformal_policy,
)
from agent.graded_decision import (
    DEFAULT_THRESHOLDS,
    decide_conformal,
    load_conformal_policy,
)
from tools.fit_conformal_policy import fit_and_validate, synthetic_rows


def test_fit_threshold_is_finite_quantile():
    rows = [{"nonconformity": i / 10, "correct": True} for i in range(10)]
    p = fit_conformal_policy(rows, alpha=0.1)
    assert isinstance(p, ConformalPolicy)
    assert 0.0 <= p.threshold <= 1.0
    assert p.target_coverage == 0.9


def test_policy_decide_is_two_way_and_monotone():
    p = ConformalPolicy(alpha=0.1, threshold=0.3, n_calibration=50, target_coverage=0.9)
    assert p.decide(0.1)["verdict"] == "answer"   # low nonconformity -> answer
    assert p.decide(0.9)["verdict"] == "abstain"  # high nonconformity -> abstain
    assert p.decide(0.3)["verdict"] == "answer"   # boundary inclusive


def test_decide_conformal_fail_closed_mapping():
    p = ConformalPolicy(alpha=0.1, threshold=0.3, n_calibration=50, target_coverage=0.9)
    # gate passed + accept -> answer ; gate passed + reject -> abstain
    assert decide_conformal(gate_passed=True, confidence=0.95, policy=p)["action"] == "answer"
    assert decide_conformal(gate_passed=True, confidence=0.2, policy=p)["action"] == "abstain"
    # gate FAILED is downgrade-only: best case is a hedge, never a full answer
    assert decide_conformal(gate_passed=False, confidence=0.95, policy=p)["action"] == "hedge"
    assert decide_conformal(gate_passed=False, confidence=0.2, policy=p)["action"] == "abstain"


def test_decide_conformal_fallback_when_no_policy():
    # With no fitted policy the threshold falls back to 1 - DEFAULT hi (a safe no-op).
    out = decide_conformal(gate_passed=True, confidence=0.95, policy=None)
    assert out["policySource"] == "fallback-default-threshold"
    assert abs(out["threshold"] - (1.0 - DEFAULT_THRESHOLDS["hi"])) < 1e-9
    assert out["coverageGuarantee"] is None


def test_load_conformal_policy_roundtrip_and_absent():
    p = fit_conformal_policy([{"nonconformity": 0.2, "correct": True}], alpha=0.1)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "policy.json"
        path.write_text(json.dumps(p.to_dict()), encoding="utf-8")
        loaded = load_conformal_policy(path)
        assert loaded is not None
        assert abs(loaded.threshold - p.threshold) < 1e-9
        assert load_conformal_policy(Path(d) / "missing.json") is None


def test_held_out_validity_holds_on_synthetic():
    # The central honest claim: the coverage guarantee holds on a held-out split.
    rows = synthetic_rows(600)
    res = fit_and_validate(rows, alpha=0.1, holdout=0.5)
    assert res["validityHolds"] is True
    for risk, body in res["byRisk"].items():
        v = body["validity"]
        if v is not None:
            assert v["holds"], f"validity failed for risk={risk}: {v}"


def test_evaluate_policy_reports_no_overclaim_fields():
    rows = synthetic_rows(100)
    p = fit_conformal_policy(rows, alpha=0.1)
    ev = evaluate_policy(p, rows)
    assert ev["candidateOnly"] is True
    assert ev["level3Evidence"] is False
    assert 0.0 <= ev["metrics"]["coverage"] <= 1.0


def test_crc_bound_holds_and_is_at_most_alpha():
    # CRC selects the largest threshold whose (n*Rhat + 1)/(n+1) <= alpha; the reported
    # bound must be <= alpha and the in-sample empirical risk must be <= the bound.
    rows = synthetic_rows(600)
    rc = conformal_risk_control(rows, alpha=0.1)
    assert rc["feasible"] is True
    assert rc["crcBound"] <= 0.1 + 1e-9
    assert rc["empiricalRisk"] <= rc["crcBound"] + 1e-9
    # in-sample realized marginal false-answer rate == empiricalRisk (by construction).
    # CRC fits on the risk_bucket subset (default "normal"), same convention as
    # fit_conformal_policy — so the realized rate is over that bucket, not all rows.
    tau = rc["threshold"]
    bucket = [r for r in rows if r.get("risk", "normal") == "normal"]
    realized = sum(1 for r in bucket if float(r["nonconformity"]) <= tau and not r["correct"]) / len(bucket)
    assert abs(realized - rc["empiricalRisk"]) < 1e-6  # empiricalRisk is stored rounded to 6dp
    assert rc["candidateOnly"] is True and rc["level3Evidence"] is False


def test_crc_infeasible_when_n_too_small():
    # alpha=0.05 needs n >= 19 (B/(n+1) <= alpha). With n=10 even abstain-all can't certify.
    rows = [{"nonconformity": i / 10, "correct": i % 2 == 0} for i in range(10)]
    rc = conformal_risk_control(rows, alpha=0.05)
    assert rc["feasible"] is False
    assert rc["threshold"] is None
    assert rc["crcBound"] > 0.05  # B/(n+1) = 1/11 ~ 0.0909


def test_crc_more_coverage_at_larger_alpha():
    # A looser risk budget answers at least as many rows (monotone in alpha).
    rows = synthetic_rows(600)
    lo = conformal_risk_control(rows, alpha=0.05)
    hi = conformal_risk_control(rows, alpha=0.2)
    assert lo["feasible"] and hi["feasible"]
    assert hi["answered"] >= lo["answered"]


def test_crc_validity_mean_realized_within_alpha():
    # The empirical demonstration of E[loss] <= alpha: mean held-out realized
    # false-answer rate across many random splits must not exceed alpha.
    rows = synthetic_rows(600)
    v = crc_validity_check(rows, alpha=0.1, n_splits=120, seed=0, holdout=0.3)
    assert v["feasibleSplits"] > 0
    assert v["meanRealizedFalseAnswerRate"] <= 0.1 + 1e-9
    assert v["valid"] is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
