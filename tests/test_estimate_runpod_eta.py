#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/estimate_runpod_eta.py (deterministic heuristic estimator)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.estimate_runpod_eta import estimate, to_markdown  # noqa: E402


def _est(mode, seeds=3, rows=754, include_eval=True):
    return estimate(model_params_b=7.0, seeds=seeds, mode=mode, epochs=2, rows=rows,
                    include_eval=include_eval)


def test_calibration_single_7b_matches_observed_range() -> None:
    # observed seed1/seed2 ~27-30 min total; point should land in a sane band
    w = _est("single", seeds=1)["wallMin"]
    assert 20 <= w["point"] <= 35
    assert w["low"] < w["point"] < w["high"]


def test_mode_ordering() -> None:
    par = _est("on-pod-parallel")["wallMin"]["point"]
    seq = _est("on-pod-sequential")["wallMin"]["point"]
    allseq = _est("all-seeds-sequential")["wallMin"]["point"]
    sep = _est("separate-pods")["wallMin"]["point"]
    # parallel (one provision + one train) is the fastest for N>1
    assert par < seq
    assert par < allseq
    # sequential-on-one-pod beats N separate full pods (no per-seed provision+eval x N)
    assert seq < allseq
    # separate-pods (concurrent) ~ one seed + eval; cheaper wall than sequential-on-pod for N=3
    assert sep < seq


def test_on_pod_modes_skip_eval() -> None:
    assert _est("on-pod-parallel")["evalMin"] == 0.0
    assert _est("on-pod-sequential")["evalMin"] == 0.0
    assert _est("separate-pods")["evalMin"] > 0.0
    # explicit train-only also zeroes eval
    assert _est("separate-pods", include_eval=False)["evalMin"] == 0.0


def test_scales_with_seeds_rows_epochs() -> None:
    base = estimate(model_params_b=7, seeds=3, mode="on-pod-sequential", epochs=2, rows=754)
    more_seeds = estimate(model_params_b=7, seeds=6, mode="on-pod-sequential", epochs=2, rows=754)
    assert more_seeds["wallMin"]["point"] > base["wallMin"]["point"]
    more_rows = estimate(model_params_b=7, seeds=3, mode="on-pod-sequential", epochs=2, rows=2000)
    assert more_rows["wallMin"]["point"] > base["wallMin"]["point"]


def test_markdown_has_eta_and_finish() -> None:
    est = _est("on-pod-parallel")
    md = to_markdown(est, datetime(2026, 6, 25, 15, 30, tzinfo=timezone.utc))
    assert "estimated wall-clock" in md
    assert "Expected finish:" in md
    assert "on-pod-parallel" in md


def main() -> int:
    test_calibration_single_7b_matches_observed_range()
    test_mode_ordering()
    test_on_pod_modes_skip_eval()
    test_scales_with_seeds_rows_epochs()
    test_markdown_has_eta_and_finish()
    print("test_estimate_runpod_eta: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
