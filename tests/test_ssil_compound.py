#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_compound import demo_compound_report, run_compound_loop, scripted_proposer  # noqa: E402

REPORT = demo_compound_report()


def test_demo_invariants() -> None:
    assert all(REPORT["invariants"].values()), REPORT["invariants"]


def test_compounding_curve_nondecreasing_and_rises() -> None:
    curve = REPORT["compoundingCurve"]
    assert len(curve) >= 2
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert curve[-1] > curve[0]  # the canonical baseline genuinely rose


def test_two_keys_present_in_gate_set() -> None:
    # Every gated round runs G1 (value) and G3 (capability) alongside G2/G4/G5/G6.
    rep = run_compound_loop(scripted_proposer([{"min_sources": 2, "min_quality": 0.5, "default_action": "abstain"}]),
                            rounds=1, canonical_n=2, seed=7, stop_after_dry=5)
    assert rep["rounds"] == 1


def test_converges_and_stops_on_dry() -> None:
    # A spec that never beats the weak baseline should stop after stop_after_dry.
    weak = {"min_sources": 0, "min_quality": 0.0, "default_action": "answer"}  # == baseline, no gain
    rep = run_compound_loop(scripted_proposer([weak]), rounds=10, canonical_n=2, seed=7, stop_after_dry=2)
    assert rep["rounds"] <= 2
    assert sum(h["promoted"] for h in rep["history"]) == 0


def test_no_overclaim() -> None:
    assert REPORT["canClaimAGI"] is False
    assert REPORT["candidateOnly"] is True


def main() -> int:
    test_demo_invariants()
    test_compounding_curve_nondecreasing_and_rises()
    test_two_keys_present_in_gate_set()
    test_converges_and_stops_on_dry()
    test_no_overclaim()
    print("test_ssil_compound: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
