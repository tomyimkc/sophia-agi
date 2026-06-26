#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke test for the offline intake candidate eval harness."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_intake_eval_sophia as intake_eval  # noqa: E402


def test_intake_eval_emits_candidate_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "intake.public-report.json"
        report = intake_eval.run(out)
        assert out.exists()
    assert report["status"] == "candidate"
    metrics = report["fidelityMetrics"]
    assert set(metrics) == {
        "intentAccuracy",
        "ambiguityRecall",
        "falseClarificationRate",
        "silentDriftRate",
        "unsurfacedRewriteRate",
    }
    assert metrics["silentDriftRate"] == 0.0
    assert metrics["unsurfacedRewriteRate"] == 0.0
    controls = report["silentDriftControls"]
    assert controls["oracle"] == "gold_meaning_tokens"
    assert controls["noDrift"]["silentDriftRate"] == 0.0
    assert controls["plantedDrift"]["silentDriftRate"] > 0.0
    assert controls["plantedDrift"]["unsurfacedRewriteRate"] == 0.0
    assert set(controls["plantedDrift"]["classes"]) == intake_eval.PLANTED_DRIFT_CLASSES
    assert "synthetic token oracle only" in controls["caveat"]
    assert report["latencyCost"]["backend"] == "deterministic_offline"
    assert report["counterfactualAudit"]["found"] is True
    assert report["executionWithinContractGate"]["gate"]["held_reason"] == "over_budget"
    assert any("synthetic gold meaning tokens" in item for item in report["measuredVsAsserted"]["measured"])
    assert any("synthetic token oracle only" in item for item in report["measuredVsAsserted"]["stillAsserted"])


def test_silent_drift_controls_are_gated() -> None:
    controls = intake_eval._drift_controls()
    assert controls["noDrift"]["silentDriftRate"] == 0.0
    assert set(controls["plantedDrift"]["classes"]) == intake_eval.PLANTED_DRIFT_CLASSES
    planted = intake_eval._score_cases(intake_runner=intake_eval._planted_drift_intake)
    planted_by_class = {
        row["category"]: row
        for row in planted["rows"]
        if row["id"] in intake_eval.PLANTED_DRIFT_REWRITES
    }
    assert set(planted_by_class) == intake_eval.PLANTED_DRIFT_CLASSES
    assert all(row["silentDrift"] for row in planted_by_class.values())
    assert not any(row["unsurfacedRewrite"] for row in planted["rows"])


def main() -> int:
    test_intake_eval_emits_candidate_report()
    test_silent_drift_controls_are_gated()
    print("test_intake_eval_sophia: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
