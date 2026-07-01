#!/usr/bin/env python3
# PLANNING/HARNESS ONLY - no capability claim; canClaimAGI stays false.
"""Standalone plain-script test for tools/eval_okf_vs_pureweight.py.

Run: python tests/test_eval_okf_vs_pureweight.py
Prints 'test_eval_okf: PASS' and exits 0 on success, else exits 1.
PURE / OFFLINE / DETERMINISTIC. stdlib only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.eval_okf_vs_pureweight import (  # noqa: E402
    compare,
    correction_cost_ratio,
    default_fixture_path,
    load_fixture,
    offline_invariants,
    path_efficiency,
    traceability_score,
)


def main():
    traces = load_fixture(default_fixture_path())
    gt = [tr["wrong_step"] for tr in traces]

    # 1) traceability finds every seeded wrong step
    assert traceability_score(traces, gt) == 1.0, "traceability must find all seeded wrong steps"

    # 2) Arm A correction_cost_ratio < 1 (cheaper than retrain); Arm B == 1.0
    assert correction_cost_ratio("A") < 1.0, "Arm A correction must be cheaper than retrain"
    assert correction_cost_ratio("B") == 1.0, "Arm B (pure-weight) is the retrain baseline"

    # 3) path_efficiency monotonic in trace length
    short = {"steps": ["s1"], "tools": ["t1"]}
    longer = {"steps": ["s1", "s2", "s3"], "tools": ["t1", "t2"]}
    assert path_efficiency(longer) > path_efficiency(short), "path_efficiency must be monotonic"

    # 4) compare() returns all 5 metrics
    table = compare(traces)
    for metric in (
        "traceability_score",
        "correction_cost_ratio",
        "path_efficiency",
        "forgetting_proxy",
        "quality_stub",
    ):
        assert metric in table, f"compare() missing metric {metric}"
        assert "A" in table[metric] and "B" in table[metric], f"{metric} missing A/B"

    # 5) determinism: same fixture -> same table twice
    assert compare(traces) == compare(traces), "compare() must be deterministic"
    # also deterministic from path
    assert compare(default_fixture_path()) == compare(default_fixture_path())

    # 6) Arm A wins traceability and correction cost; weight-control wins nothing it can't trace
    assert table["traceability_score"]["A"] > table["traceability_score"]["B"], (
        "Arm A must out-trace the opaque weight control"
    )
    assert table["correction_cost_ratio"]["A"] < table["correction_cost_ratio"]["B"], (
        "Arm A correction must be cheaper than Arm B"
    )

    # 7) harness's own offline invariants pass
    ok, checks = offline_invariants()
    assert ok, f"offline_invariants failed: {checks}"

    print("test_eval_okf: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"test_eval_okf: FAIL - {exc}")
        sys.exit(1)
