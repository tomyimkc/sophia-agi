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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
