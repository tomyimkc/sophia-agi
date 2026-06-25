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
    }
    assert report["latencyCost"]["backend"] == "deterministic_offline"
    assert report["counterfactualAudit"]["found"] is True
    assert report["executionWithinContractGate"]["gate"]["held_reason"] == "over_budget"


def main() -> int:
    test_intake_eval_emits_candidate_report()
    print("test_intake_eval_sophia: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
