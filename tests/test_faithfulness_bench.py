#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the CoT faithfulness benchmark (C4)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_faithfulness_bench import build_report, run_drop_discrimination


def test_drop_separates_loadbearing_from_decorative():
    # The central claim: load-bearing CoT shows a larger gold-logprob drop than
    # decorative CoT under reasoning-only perturbation.
    disc = run_drop_discrimination()
    assert disc["loadBearingMeanDrop"] is not None
    assert disc["decorativeMeanDrop"] is not None
    assert disc["loadBearingMeanDrop"] > disc["decorativeMeanDrop"]
    assert disc["separation"] > 0
    # decorative CoT should barely move the gold answer (~0 drop)
    assert abs(disc["decorativeMeanDrop"]) < 0.1


def test_auroc_perfect_separation_on_fixture():
    disc = run_drop_discrimination()
    assert disc["auroc"] == 1.0  # the fixture is cleanly separable by construction


def test_cross_trace_finds_planted_contradiction():
    report = build_report()
    ct = report["crossTrace"]
    assert len(ct["contradictions"]) >= 1
    assert ct["globalConsistent"] is False


def test_report_no_overclaim_fields():
    report = build_report()
    assert report["candidateOnly"] is True
    assert report["level3Evidence"] is False
    assert report["validated"] is False
    assert report["syntheticData"] is True
    assert "FALSIFIED" in report["honestBound"]  # owns the v1 falsification


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
