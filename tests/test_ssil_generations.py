#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_generations import (  # noqa: E402
    _gen,
    compounding_proof,
    demo_compounding_report,
    evaluate_generations,
)


def test_monotone_rising_curve() -> None:
    gens = [
        _gen(1, "base", [0.70, 0.71, 0.72], before=0.53),
        _gen(2, "B1", [0.78, 0.79, 0.80], before=0.71),
        _gen(3, "B2", [0.84, 0.85, 0.86], before=0.79),
    ]
    r = evaluate_generations(gens, gated=True)
    assert r["curve"] == [0.71, 0.79, 0.85]
    assert r["monotoneRising"] is True
    assert r["convergedAt"] is None


def test_converges_when_no_gain() -> None:
    gens = [
        _gen(1, "base", [0.70, 0.71, 0.72], before=0.53),
        _gen(2, "B1", [0.70, 0.71, 0.72], before=0.71),  # same as canonical -> no gain
    ]
    r = evaluate_generations(gens, gated=True)
    assert r["curve"] == [0.71]
    assert r["convergedAt"] == 2


def test_overlapping_ci_does_not_compound() -> None:
    # A tiny gain within noise (wide spread) must NOT count as a generation step.
    gens = [
        _gen(1, "base", [0.69, 0.71, 0.73], before=0.53),
        _gen(2, "B1", [0.70, 0.72, 0.74], before=0.71),  # mean +0.01, CIs overlap
    ]
    r = evaluate_generations(gens, gated=True, min_delta=0.005)
    assert len(r["curve"]) == 1  # gen 2 rejected on CI overlap
    assert r["convergedAt"] == 2


def test_gate_rejects_contamination_but_control_admits() -> None:
    gens = [
        _gen(1, "base", [0.70, 0.71, 0.72], before=0.53),
        _gen(2, "B1", [0.95, 0.96, 0.97], before=0.71, contaminated=True),  # gamed
    ]
    proof = compounding_proof(gens)
    assert 2 in proof["gateCaughtGenerations"]
    assert 2 in {r["gen"] for r in proof["negativeControl"]["generations"] if r["promoted"]}
    assert proof["gateMadeADifference"] is True
    assert proof["gated"]["curve"] == [0.71]  # contaminated gen not in the gated curve


def test_protected_regression_blocks_generation() -> None:
    g1 = _gen(1, "base", [0.70, 0.71, 0.72], before=0.53)
    g2 = _gen(2, "B1", [0.84, 0.85, 0.86], before=0.71, protected=0.79, protected_after=0.50)  # integrity dropped 0.79->0.50
    r = evaluate_generations([g1, g2], gated=True)
    assert r["curve"] == [0.71]  # gen 2 blocked despite higher capability
    assert r["convergedAt"] == 2


def test_demo_invariants() -> None:
    rep = demo_compounding_report()
    assert all(rep["invariants"].values()), rep["invariants"]
    assert rep["canClaimAGI"] is False


def main() -> int:
    test_monotone_rising_curve()
    test_converges_when_no_gain()
    test_overlapping_ci_does_not_compound()
    test_gate_rejects_contamination_but_control_admits()
    test_protected_regression_blocks_generation()
    test_demo_invariants()
    print("test_ssil_generations: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
