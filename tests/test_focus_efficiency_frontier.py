# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Focus Efficiency Frontier harness — NO-GO by design + committed-artifact drift."""
from __future__ import annotations

import json
from pathlib import Path

from tools.run_focus_efficiency_frontier import REPORT, build_report

ROOT = Path(__file__).resolve().parents[1]


def test_report_is_nogo_by_design():
    r = build_report()
    assert r["verdict"] == "NO-GO"
    assert r["go"] is False
    assert r["canClaimAGI"] is False
    assert r["delta"] is None
    assert len(r["criticalFailures"]) >= 4


def test_routing_fidelity_perfect_on_author_battery():
    r = build_report()
    assert r["routingFidelity"]["routingAccuracy"] == 1.0


def test_mechanism_anchored_admits_fewer_offgoal_tokens():
    m = build_report()["packingMechanism"]
    assert m["prosocheAnchoredOffGoalTokens"] < m["priorityPackedOffGoalTokens"]
    assert m["anchorIsCacheStable"] is True


def test_committed_artifact_matches_fresh_build():
    # The committed PENDING report must not drift from a fresh deterministic build.
    assert REPORT.exists(), "run `python tools/run_focus_efficiency_frontier.py --write`"
    on_disk = json.loads(REPORT.read_text(encoding="utf-8"))
    fresh = build_report()
    assert on_disk["verdict"] == fresh["verdict"]
    assert on_disk["criticalFailures"] == fresh["criticalFailures"]
    assert on_disk["routingFidelity"]["routingAccuracy"] == fresh["routingFidelity"]["routingAccuracy"]


def test_measurement_spec_has_guardrails():
    spec = json.loads((ROOT / "agi-proof/benchmark-results/prosoche/measurement_spec.json").read_text())
    g = spec["guardrails"]
    # the two guardrails that make the efficiency claim honest + the safety floor
    assert "taskSuccessGuardrail" in g
    assert "antiFixationGuardrail" in g
    assert "safetyFloor" in g
    assert spec["canClaimAGI"] is False
