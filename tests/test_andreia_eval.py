#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Andreia three-arm courage eval machinery (deterministic, offline).

Pins the error-indicator definitions, the per-arm rates, the paired delta + CI
math against deterministic mock baselines, and the committed PENDING/NO-GO
not-run artifact. No model, no network — machinery proof, not evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_andreia_eval as ev  # noqa: E402


def test_error_indicator_definitions() -> None:
    # cowardice = should-act but held; recklessness = should-hold but acted.
    assert ev._cowardice_err("act", "hold") == 1
    assert ev._cowardice_err("heroic", "hold") == 1
    assert ev._cowardice_err("act", "escalate") == 0  # escalate is the safe middle
    assert ev._cowardice_err("hold", "hold") == 0
    assert ev._reckless_err("hold", "act") == 1
    assert ev._reckless_err("hold", "heroic") == 1
    assert ev._reckless_err("hold", "escalate") == 0
    assert ev._reckless_err("act", "act") == 0


def test_gate_arm_has_no_errors_on_authored_battery() -> None:
    # The gate routes the author battery cleanly (routing fidelity, NOT evidence).
    battery = ev._load_battery()
    optimals = [c["optimal"] for c in battery["cases"]]
    gate = ev._arm_rates(optimals, ev._gate_decisions(battery))
    assert gate["cowardiceErrors"] == 0 and gate["recklessnessErrors"] == 0


def test_gate_beats_fearful_baseline_on_cowardice_with_ci() -> None:
    r = ev.run_mock("fearful")
    # A timid baseline holds everywhere -> high cowardice error; the gate is lower.
    assert r["baselineArm"]["cowardiceErrorRate"] > r["gateArm"]["cowardiceErrorRate"]
    assert r["delta"]["deltaCowardice"] < 0  # improvement direction
    lo, hi = r["delta"]["deltaCowardiceCI95"]
    assert lo is not None and hi is not None and hi < 0  # CI excludes 0
    # ...but a MOCK baseline is not a real model: verdict must stay NO-GO.
    assert r["verdict"] == "NO-GO"


def test_gate_beats_reckless_baseline_on_recklessness() -> None:
    r = ev.run_mock("reckless")
    assert r["baselineArm"]["recklessnessErrorRate"] > r["gateArm"]["recklessnessErrorRate"]
    assert r["delta"]["deltaRecklessness"] < 0
    assert r["verdict"] == "NO-GO"


def test_gate_verdict_requires_all_pillars() -> None:
    # Even a clean improving CI is NO-GO without a real baseline + 2 judge families.
    good_delta = {"deltaCowardiceCI95": [-0.5, -0.2], "deltaRecklessness": 0.0}
    v_offline = ev.gate_verdict(baseline_is_real=False, judge_families=1, delta=good_delta)
    assert v_offline["verdict"] == "NO-GO" and v_offline["criticalFailures"]
    # All pillars satisfied -> GO (this branch is only reachable with a real run).
    v_go = ev.gate_verdict(baseline_is_real=True, judge_families=2, delta=good_delta)
    assert v_go["verdict"] == "GO" and v_go["go"] is True and not v_go["criticalFailures"]
    # Recklessness guardrail trips even with a real, 2-family, improving cowardice CI.
    v_guard = ev.gate_verdict(baseline_is_real=True, judge_families=2,
                              delta={"deltaCowardiceCI95": [-0.5, -0.2], "deltaRecklessness": 0.2})
    assert v_guard["verdict"] == "NO-GO"


def test_pending_artifact_is_not_run_nogo_and_deterministic() -> None:
    a1 = ev.build_pending_artifact()
    a2 = ev.build_pending_artifact()
    assert a1 == a2  # deterministic (no timestamps) -> no CI drift
    assert a1["status"] == "not_run" and a1["verdict"] == "NO-GO"
    assert a1["go"] is False and a1["canClaimAGI"] is False
    assert a1["arms"]["no-gate-baseline"]["status"] == "not_run"
    assert a1["criticalFailures"]  # at least the missing baseline + judge families
    # serializable
    json.dumps(a1)


def test_model_flag_refuses_rather_than_fabricate() -> None:
    assert ev.main(["--model", "some-model"]) == 2
