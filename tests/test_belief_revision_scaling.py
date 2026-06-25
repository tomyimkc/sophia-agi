#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.belief_revision_scaling — the revision cost-curve harness.

Verifies the synthetic graph produces the expected conflicts and the revise-or-abstain
policy retracts the right (lower-tier) losers, and that a sweep returns one measurement
per size. Correctness is asserted; wall-time is only sanity-checked (>= 0), since timing
is environment-dependent. Offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.belief_revision_scaling import make_pages, measure, scaling_sweep  # noqa: E402
from agent.belief_revision_policy import resolve_conflicts  # noqa: E402


def test_synthetic_graph_conflicts_resolve_correctly() -> None:
    # 30 facts, a contradiction every 10 -> conflicts at f10/f20 (f0 has none: i>0 guard).
    pages = make_pages(30, contradiction_every=10)
    report = resolve_conflicts(pages)
    assert report["conflictCount"] == 2
    # the axiom-tier f10/f20 win; the predecessors f9/f19 are retracted
    assert "f9" in report["retracted"] and "f19" in report["retracted"]
    assert "f10" in report["kept"] and "f20" in report["kept"]


def test_measure_and_sweep_shape() -> None:
    m = measure(50)
    assert m["n"] == 50 and m["conflicts"] >= 1 and m["seconds"] >= 0.0
    sweep = scaling_sweep([20, 40, 80])
    assert [r["n"] for r in sweep] == [20, 40, 80]
    assert all(r["seconds"] >= 0.0 for r in sweep)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
