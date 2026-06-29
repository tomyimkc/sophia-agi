#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""T5: intelligence-per-parameter/per-byte as a measured axis with CIs + Pareto frontier."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_efficiency_frontier as ef  # noqa: E402


def _entries():
    return [
        {"label": "small-4b", "score": 0.80, "scoreMetric": "passAt1", "n": 200,
         "activeParamsB": 4.0, "servedGB": 7.0},
        {"label": "big-70b", "score": 0.82, "scoreMetric": "passAt1", "n": 200,
         "activeParamsB": 70.0, "servedGB": 40.0},
    ]


def test_per_param_efficiency_and_ci() -> None:
    f = ef.build_frontier(_entries())
    by = {e["label"]: e for e in f["entries"]}
    # 0.80/4.0 = 0.2 per active-B; the bigger model is far less efficient per param.
    assert by["small-4b"]["perActiveParam"]["value"] == 0.2
    assert by["big-70b"]["perActiveParam"]["value"] == round(0.82 / 70.0, 4)
    # a real CI is attached (proportion metric + n) and brackets the point estimate
    ci = by["small-4b"]["scoreCI"]
    assert ci is not None and ci[0] < 0.80 < ci[1]
    pp_ci = by["small-4b"]["perActiveParam"]["ci"]
    assert pp_ci[0] < 0.2 < pp_ci[1]


def test_ranking_puts_most_efficient_first() -> None:
    f = ef.build_frontier(_entries())
    assert f["entries"][0]["label"] == "small-4b"  # highest score/B


def test_pareto_frontier() -> None:
    # both entries in _entries() are Pareto-optimal: big-70b wins on score (0.82 > 0.80),
    # small-4b wins on both costs — a genuine trade-off, neither dominates.
    f = ef.build_frontier(_entries())
    assert set(f["paretoOptimal"]) == {"small-4b", "big-70b"}
    # now make the small model strictly better on score too -> it dominates the big one.
    dominated = [
        {"label": "small-4b", "score": 0.85, "scoreMetric": "passAt1", "n": 200,
         "activeParamsB": 4.0, "servedGB": 7.0},
        {"label": "big-70b", "score": 0.82, "scoreMetric": "passAt1", "n": 200,
         "activeParamsB": 70.0, "servedGB": 40.0},
    ]
    f2 = ef.build_frontier(dominated)
    assert f2["paretoOptimal"] == ["small-4b"]


def test_missing_costs_are_tolerated() -> None:
    f = ef.build_frontier([{"label": "x", "score": 0.5, "scoreMetric": "meanReward"}])
    e = f["entries"][0]
    assert e["perActiveParam"]["value"] is None
    assert e["scoreCI"] is None  # meanReward is not a proportion metric here


def main() -> int:
    test_per_param_efficiency_and_ci()
    test_ranking_puts_most_efficient_first()
    test_pareto_frontier()
    test_missing_costs_are_tolerated()
    print("test_efficiency_frontier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
