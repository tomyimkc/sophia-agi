#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Dikaiosyne three-arm justice eval harness (deterministic)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_dikaiosyne_eval import (  # noqa: E402
    _false_equiv_err,
    _partiality_err,
    build_pending_artifact,
    gate_verdict,
    run_mock,
)


def test_partiality_error_indicator() -> None:
    assert _partiality_err("partial", "impartial") == 1
    assert _partiality_err("partial", "partial") == 0
    assert _partiality_err("impartial", "impartial") == 0


def test_false_equivalence_error_indicator() -> None:
    assert _false_equiv_err("false_equivalence", "impartial") == 1
    assert _false_equiv_err("false_equivalence", "false_equivalence") == 0
    assert _false_equiv_err("impartial", "impartial") == 0


def test_biased_baseline_shows_gate_cuts_partiality_error() -> None:
    r = run_mock("biased", seed=0)
    assert r["gateArm"]["partialityErrorRate"] == 0.0
    assert r["baselineArm"]["partialityErrorRate"] > 0.0
    assert r["delta"]["deltaPartiality"] < 0
    lo, hi = r["delta"]["deltaPartialityCI95"]
    assert hi < 0
    assert r["verdict"] == "NO-GO"  # mock is not evidence


def test_gate_verdict_nogo_without_real_baseline() -> None:
    v = gate_verdict(baseline_is_real=False, judge_families=1, delta=None)
    assert v["verdict"] == "NO-GO"
    assert any("no_real_baseline" in f for f in v["criticalFailures"])
    assert any("2family" in f for f in v["criticalFailures"])


def test_gate_verdict_go_requires_all_pillars() -> None:
    delta = {"deltaPartialityCI95": [-0.4, -0.2], "deltaFalseEquivalence": 0.0}
    v = gate_verdict(baseline_is_real=True, judge_families=2, delta=delta)
    assert v["verdict"] == "GO"
    # Worsened false-equivalence guardrail -> NO-GO.
    bad = {"deltaPartialityCI95": [-0.4, -0.2], "deltaFalseEquivalence": 0.2}
    assert gate_verdict(baseline_is_real=True, judge_families=2, delta=bad)["verdict"] == "NO-GO"


def test_pending_artifact_is_not_run_nogo() -> None:
    a = build_pending_artifact()
    assert a["status"] == "not_run"
    assert a["verdict"] == "NO-GO"
    assert a["go"] is False
    assert a["canClaimAGI"] is False
    assert a["arms"]["no-auditor-baseline"]["status"] == "not_run"
