# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Focus-Efficiency-Frontier 3-arm eval — mechanism, guardrail math, gate, determinism."""
from __future__ import annotations

import json

from tools.run_focus_frontier_eval import (
    ARMS,
    PENDING_PATH,
    build_battery,
    build_pending_artifact,
    gate_verdict,
    run_mock,
)


def test_three_real_arms_run():
    r = run_mock(seed=0)
    assert set(r["arms"]) == set(ARMS)
    for a in ARMS:
        assert r["arms"][a]["n"] == len(build_battery())


def test_anchored_arm_detects_the_efficiency_direction():
    # The survival-proxy mock must show the harness CAN detect the effect direction:
    # the anchored arm is more token-efficient (negative delta) with a CI excluding 0.
    r = run_mock(seed=0)
    pd = r["primaryDelta"]
    assert pd["deltaEfficiencyCost"] < 0
    lo, hi = pd["deltaTokensCI95"]
    assert lo is not None and hi is not None and hi < 0  # CI excludes 0 on the fewer-tokens side


def test_guardrails_hold_in_mock():
    g = run_mock(seed=0)["guardrails"]
    assert g["successNonInferior"] is True
    assert g["antiFixationHeld"] is True
    assert g["safetyPruned"] == 0


def test_mock_is_nogo_only_for_model_gating():
    # The math/guardrails pass; the ONLY reasons it is NO-GO are the model-gating ones.
    r = run_mock(seed=0)
    assert r["verdict"] == "NO-GO"
    joined = " ".join(r["criticalFailures"])
    assert "no_real_arms" in joined and "ground_truth_not_2family" in joined
    assert "task_success_regressed" not in joined
    assert "safety_pruned" not in joined


def test_gate_go_reachable_when_all_conditions_met():
    go = gate_verdict(baseline_is_real=True, judge_families=2,
                      delta={"deltaTokensCI95": [-112.0, -37.0], "deltaTokensCS95": [-100.0, -20.0]},
                      success_guardrail_held=True, antifixation_held=True, safety_pruned=0)
    assert go["verdict"] == "GO" and go["go"] is True


def test_gate_safety_prune_forces_nogo():
    v = gate_verdict(baseline_is_real=True, judge_families=2,
                     delta={"deltaTokensCI95": [-112.0, -37.0], "deltaTokensCS95": [-100.0, -20.0]},
                     success_guardrail_held=True, antifixation_held=True, safety_pruned=1)
    assert v["verdict"] == "NO-GO"
    assert any("safety_pruned" in f for f in v["criticalFailures"])


def test_gate_positive_ci_is_nogo():
    # A CI that does NOT exclude 0 on the negative side cannot be a GO.
    v = gate_verdict(baseline_is_real=True, judge_families=2,
                     delta={"deltaTokensCI95": [-10.0, 5.0]},
                     success_guardrail_held=True, antifixation_held=True, safety_pruned=0)
    assert v["verdict"] == "NO-GO"


def test_gate_requires_anytime_cs_too():
    # Bootstrap CI excludes 0 but the (more conservative) anytime-valid CS does not -> NO-GO.
    v = gate_verdict(baseline_is_real=True, judge_families=2,
                     delta={"deltaTokensCI95": [-50.0, -10.0], "deltaTokensCS95": [-40.0, 5.0]},
                     success_guardrail_held=True, antifixation_held=True, safety_pruned=0)
    assert v["verdict"] == "NO-GO"
    assert any("anytime_cs" in f for f in v["criticalFailures"])


def test_pending_artifact_is_nogo_and_deterministic():
    a = build_pending_artifact()
    b = build_pending_artifact()
    assert a == b  # no timestamps / RNG drift
    assert a["verdict"] == "NO-GO" and a["canClaimAGI"] is False
    assert a["delta"] is None


def test_committed_artifact_matches_fresh_build():
    assert PENDING_PATH.exists(), "run `python tools/run_focus_frontier_eval.py --write`"
    on_disk = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    fresh = build_pending_artifact()
    assert on_disk["verdict"] == fresh["verdict"]
    assert on_disk["primaryDeltaProxy"] == fresh["primaryDeltaProxy"]
    assert on_disk["criticalFailures"] == fresh["criticalFailures"]
