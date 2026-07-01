# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prosoche robustness probe — explicit anchor vs derived goal (honest gap, NO-GO)."""
from __future__ import annotations

import json
from pathlib import Path

from tools.run_prosoche_robustness import OUT, run

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_at_least_as_good_as_derived():
    r = run()
    # The whole point: an explicit anchor routes at least as well as a derived goal.
    assert r["explicitAccuracy"] >= r["derivedAccuracy"]
    assert r["gap"] == round(r["explicitAccuracy"] - r["derivedAccuracy"], 4)


def test_gap_is_reported_not_zero_by_construction():
    # The probe must actually exercise the weakness — a positive gap is expected, and
    # it must be reported (the ledger row depends on this being surfaced, not hidden).
    r = run()
    assert r["gap"] > 0.0
    assert r["canClaimAGI"] is False
    assert "model-gated" in r["finding"]


def test_committed_artifact_matches_fresh_run():
    assert OUT.exists(), "run `python tools/run_prosoche_robustness.py --write`"
    on_disk = json.loads(OUT.read_text(encoding="utf-8"))
    fresh = run()
    assert on_disk["explicitAccuracy"] == fresh["explicitAccuracy"]
    assert on_disk["derivedAccuracy"] == fresh["derivedAccuracy"]
