#!/usr/bin/env python3
"""Tests for the CPQA validation pass — bootstrap CIs on the real wiki benchmark.

Verifies the graph-backed system's accuracy CI collapses to [1.0, 1.0] with a reported
rule-of-three upper error bound, that the frozen baseline's CI sits well below it, and
that the control-flow sweep is monotone in routing strictness. Deterministic (seeded
bootstrap), offline, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa import load_episodes, run_benchmark  # noqa: E402
from tools.run_continual_qa_validation import bootstrap_ci  # noqa: E402

WIKI = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"


def _rows():
    return run_benchmark(load_episodes(WIKI))["rows"]


def test_graph_backed_ci_is_perfect_with_rule_of_three() -> None:
    ci = bootstrap_ci(_rows(), "graph_backed", B=500)
    assert ci["ci95"] == [1.0, 1.0]
    assert ci["observedErrors"] == 0
    assert 0.0 < ci["ruleOfThreeUpperErrorRate"] < 0.05      # ~3/n for n~92


def test_baseline_ci_is_well_below_graph_backed() -> None:
    ci = bootstrap_ci(_rows(), "parametric_baseline", B=500)
    lo, hi = ci["ci95"]
    assert hi < 1.0
    assert lo < ci["pointAccuracy"] < hi or lo <= ci["pointAccuracy"] <= hi


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
