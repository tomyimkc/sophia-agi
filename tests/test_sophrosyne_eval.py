#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Sophrosyne three-arm temperance eval harness (deterministic).

Mirrors test_andreia_eval: pins the error-indicator logic, the per-arm rates, the
paired delta+CI on deterministic mock baselines, and the committed PENDING/NO-GO
artifact. A mock is NOT a model — the verdict must stay NO-GO regardless.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_sophrosyne_eval import (  # noqa: E402
    _deficiency_err,
    _excess_err,
    build_pending_artifact,
    gate_verdict,
    run_mock,
)


def test_excess_error_indicator() -> None:
    # restrained when the optimal was to keep/spend -> excess error.
    assert _excess_err("sustain", "restrain") == 1
    assert _excess_err("proportionate", "restrain") == 1
    assert _excess_err("restrain", "restrain") == 0
    assert _excess_err("sustain", "sustain") == 0


def test_deficiency_error_indicator() -> None:
    assert _deficiency_err("restrain", "sustain") == 1
    assert _deficiency_err("proportionate", "sustain") == 1
    assert _deficiency_err("sustain", "sustain") == 0
    assert _deficiency_err("restrain", "restrain") == 0


def test_escalate_is_neither_error() -> None:
    assert _excess_err("restrain", "escalate") == 0
    assert _deficiency_err("sustain", "escalate") == 0


def test_miserly_baseline_shows_gate_cuts_excess_error() -> None:
    # A miserly baseline (always restrain) has high excess error; the gate's is 0,
    # so the paired delta is negative and its CI excludes 0 — but a mock is not a model.
    r = run_mock("miserly", seed=0)
    assert r["gateArm"]["excessErrorRate"] == 0.0
    assert r["baselineArm"]["excessErrorRate"] > 0.0
    assert r["delta"]["deltaExcess"] < 0
    lo, hi = r["delta"]["deltaExcessCI95"]
    assert hi < 0  # improvement CI strictly below zero
    assert r["verdict"] == "NO-GO"  # mock is not evidence


def test_profligate_baseline_shows_gate_cuts_deficiency_error() -> None:
    r = run_mock("profligate", seed=0)
    assert r["gateArm"]["deficiencyErrorRate"] == 0.0
    assert r["baselineArm"]["deficiencyErrorRate"] > 0.0
    assert r["delta"]["deltaDeficiency"] < 0
    assert r["verdict"] == "NO-GO"


def test_gate_verdict_is_nogo_without_real_baseline() -> None:
    v = gate_verdict(baseline_is_real=False, judge_families=1, delta=None)
    assert v["verdict"] == "NO-GO"
    assert any("no_real_baseline" in f for f in v["criticalFailures"])
    assert any("2family" in f for f in v["criticalFailures"])


def test_gate_verdict_go_requires_all_pillars() -> None:
    # Even with a perfect delta, GO needs a real baseline, 2 families, and the guardrail.
    delta = {"deltaExcessCI95": [-0.4, -0.2], "deltaDeficiencyCI95": [-0.4, -0.2]}
    v = gate_verdict(baseline_is_real=True, judge_families=2, delta=delta,
                     task_success_guardrail_measured=True)
    assert v["verdict"] == "GO"
    # Drop any single pillar -> NO-GO.
    assert gate_verdict(baseline_is_real=True, judge_families=2, delta=delta,
                        task_success_guardrail_measured=False)["verdict"] == "NO-GO"


def test_pending_artifact_is_not_run_nogo() -> None:
    a = build_pending_artifact()
    assert a["status"] == "not_run"
    assert a["verdict"] == "NO-GO"
    assert a["go"] is False
    assert a["canClaimAGI"] is False
    assert a["arms"]["no-gate-baseline"]["status"] == "not_run"
