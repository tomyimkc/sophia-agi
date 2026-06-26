#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.quality_regression (Phase 3), the fail-closed gate.

Verifies that an identical/improved corpus passes, that a mean-quality drop, a keep-rate
drop, a duplicate-rate spike, and a large token-volume loss each fail, that small changes
within tolerance pass, and that a vanished meanQuality fails closed. Pure stdlib.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.quality_regression import Tolerances, compare, gate  # noqa: E402

_BASE = {
    "meanQuality": 0.80,
    "keepRate": 0.70,
    "duplicateRate": 0.05,
    "totalTokens": 1000,
}


def test_identical_passes():
    assert compare(_BASE, dict(_BASE)) == []
    assert gate(_BASE, dict(_BASE))["ok"] is True


def test_improvement_passes():
    better = {"meanQuality": 0.9, "keepRate": 0.8, "duplicateRate": 0.01, "totalTokens": 2000}
    assert compare(_BASE, better) == []


def test_quality_drop_fails():
    worse = dict(_BASE, meanQuality=0.70)  # -0.10 > 0.05 tol
    problems = compare(_BASE, worse)
    assert any("meanQuality" in p for p in problems)
    assert gate(_BASE, worse)["ok"] is False


def test_keep_rate_drop_fails():
    worse = dict(_BASE, keepRate=0.60)  # -0.10
    assert any("keepRate" in p for p in compare(_BASE, worse))


def test_duplicate_spike_fails():
    worse = dict(_BASE, duplicateRate=0.20)  # +0.15
    assert any("duplicateRate" in p for p in compare(_BASE, worse))


def test_token_collapse_fails():
    worse = dict(_BASE, totalTokens=100)  # -90% > 50% tol
    assert any("totalTokens" in p for p in compare(_BASE, worse))


def test_within_tolerance_passes():
    ok = dict(_BASE, meanQuality=0.78, keepRate=0.68, duplicateRate=0.07, totalTokens=900)
    assert compare(_BASE, ok) == []


def test_custom_tolerance():
    worse = dict(_BASE, meanQuality=0.78)  # -0.02
    strict = Tolerances(max_quality_drop=0.01)
    assert any("meanQuality" in p for p in compare(_BASE, worse, tol=strict))


def test_vanished_quality_fails_closed():
    worse = dict(_BASE, meanQuality=None)
    assert any("meanQuality" in p for p in compare(_BASE, worse))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.quality_regression tests passed")
