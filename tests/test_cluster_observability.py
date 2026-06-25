#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Observability stats + straggler analysis tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.observability import percentile, straggler_report, summarize


def test_percentile_basic() -> None:
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 0) == 1
    assert percentile(xs, 100) == 10
    assert abs(percentile(xs, 50) - 5.5) < 1e-9
    assert percentile([], 50) == 0.0
    assert percentile([42], 99) == 42


def test_summary_jitter_cv() -> None:
    flat = summarize([10.0] * 8)
    assert flat.cv == 0.0  # no jitter
    assert flat.p99 == flat.p50 == 10.0
    noisy = summarize([10, 10, 10, 10, 50])  # one spike
    assert noisy.cv > 0.0
    assert noisy.max == 50
    assert noisy.p99 > noisy.p50


def test_straggler_report_identifies_long_pole() -> None:
    # rank 3 is the straggler
    per_rank = [1.0, 1.0, 1.0, 2.0]
    rep = straggler_report(per_rank)
    assert rep.slowest_rank == 3
    assert rep.fastest_rank in (0, 1, 2)
    # a synchronous step waits for the slowest: slowdown = max/mean = 2.0/1.25 = 1.6
    assert abs(rep.step_slowdown - 1.6) < 1e-9
    assert rep.skew > 0.0


def test_straggler_balanced_is_neutral() -> None:
    rep = straggler_report([3.0, 3.0, 3.0, 3.0])
    assert abs(rep.step_slowdown - 1.0) < 1e-9
    assert rep.skew == 0.0
    assert abs(rep.tail_ratio - 1.0) < 1e-9


def main() -> int:
    test_percentile_basic()
    test_summary_jitter_cv()
    test_straggler_report_identifies_long_pole()
    test_straggler_balanced_is_neutral()
    print("test_cluster_observability: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
