# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/stage_decomposition_report.py (A7 reporting patterns)."""
from __future__ import annotations

import pytest

from tools.stage_decomposition_report import baseline_provenance, build_stage_decomposition


def test_regressions_are_fail_visible():
    stages = [
        ("base", {"gaia": 59.8, "hle": 47.4}),
        ("sft", {"gaia": 95.2, "hle": 41.6}),     # the paper's HLE -5.8 pattern
        ("final", {"gaia": 96.0, "hle": 47.6}),
    ]
    out = build_stage_decomposition(stages)
    assert out["stages"] == ["base", "sft", "final"]
    regs = {(r["benchmark"], r["stage"]): r["delta"] for r in out["regressions"]}
    assert regs == {("hle", "sft"): -5.8}, "the SFT-stage HLE drop must be surfaced"
    gaia = next(r for r in out["rows"] if r["benchmark"] == "gaia")
    assert gaia["deltas"] == [pytest.approx(35.4), pytest.approx(0.8)]


def test_missing_benchmark_yields_none_not_crash_and_min_stages():
    out = build_stage_decomposition([("base", {"a": 1.0}), ("final", {"b": 2.0})])
    assert {r["benchmark"]: r["deltas"] for r in out["rows"]} == {"a": [None], "b": [None]}
    with pytest.raises(ValueError):
        build_stage_decomposition([("only", {"a": 1.0})])


def test_baseline_provenance_dual_report():
    bp = baseline_provenance(official=81.2, reproduced=32.5, tolerance=1.0, source="tau2")
    assert bp["discrepant"] is True and bp["useForClaims"] == 32.5
    ok = baseline_provenance(official=50.0, reproduced=49.5, tolerance=1.0)
    assert ok["discrepant"] is False and ok["useForClaims"] == 49.5
    absent = baseline_provenance(official=50.0, reproduced=None)
    assert absent["useForClaims"] is None, "no reproduced number -> nothing claimable"
