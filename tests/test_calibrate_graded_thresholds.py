#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the graded-threshold calibration tool (S3).

Covers the offline operating curve (answer-coverage decreases monotonically as hi rises;
current default reported; candidate-marked), the live-signal -> dataset loop-closer
(records carry a live confidence), and the production fit path over labeled outcomes.
Deterministic, offline; the fit reuses the existing calibrate() so defaults are never baked.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calibrate_graded_thresholds import (  # noqa: E402
    emit_records, fit_from_data, operating_curve,
)


def test_operating_curve_is_monotonic_and_candidate():
    report = operating_curve()
    assert report["candidateOnly"] is True and report["validated"] is False
    assert report["n"] > 0
    answers = [p["answer"] for p in report["curve"]]
    # Raising hi can only move pages out of "answer" -> coverage is non-increasing.
    assert all(answers[i] >= answers[i + 1] for i in range(len(answers) - 1))
    # Each operating point's mix is a partition.
    for p in report["curve"]:
        assert abs(p["answer"] + p["hedge"] + p["abstain"] - 1.0) < 1e-6
    assert "mix" in report["currentDefault"]


def test_loop_closer_emits_live_confidence_records():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "runs.jsonl"
        summary = emit_records(out)
        assert summary["emitted"] > 0 and summary["withConfidence"] > 0
        rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert rows and all("confidence" in r and "policy" in r for r in rows)


def test_fit_from_labeled_data_reports_candidate_not_baked():
    # A separable labeled set: correct answers high-confidence, errors low.
    records = ([{"confidence": 0.9, "correct": True}] * 6
               + [{"confidence": 0.2, "correct": False}] * 6)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "labeled.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        result = fit_from_data(p)
    assert 0.0 <= result["bestHi"] <= 1.0 and 0.0 <= result["lo"] <= 1.0
    assert result["balancedAccuracy"] == 1.0
    assert result["currentDefault"] == {"hi": 0.7, "lo": 0.4}  # unchanged reference
    assert "CANDIDATE" in result["note"]


def test_defaults_are_not_mutated():
    from agent.graded_decision import DEFAULT_THRESHOLDS

    operating_curve()
    assert DEFAULT_THRESHOLDS == {"hi": 0.7, "lo": 0.4}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
