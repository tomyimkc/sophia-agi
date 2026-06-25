#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.forgetting_curve — graceful, measurable decay.

Verifies retention is in [0,1] and monotone (down with disuse, up with reinforcement),
and that a stale unused fact fades below threshold while a reinforced one survives.
Deterministic math; offline, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.forgetting_curve import faded, forgetting_report, retention  # noqa: E402


def test_retention_bounds_and_monotonicity() -> None:
    assert retention(0, 0) == 1.0                                  # just seen -> full
    assert 0.0 < retention(90, 0) < 1.0
    assert retention(90, 0) < retention(30, 0)                     # more disuse -> less retention
    assert retention(90, 5) > retention(90, 0)                     # reinforcement stabilizes


def test_stale_fact_fades_reinforced_survives() -> None:
    facts = [
        {"id": "stale", "rank": 4, "daysSince": 400, "reinforcements": 0},
        {"id": "fresh", "rank": 4, "daysSince": 5, "reinforcements": 0},
        {"id": "reinforced", "rank": 4, "daysSince": 400, "reinforcements": 10},
    ]
    rep = forgetting_report(facts, assert_threshold=1.0)
    assert "stale" in rep["faded"]
    assert "fresh" not in rep["faded"]
    assert "reinforced" not in rep["faded"]                        # reinforcement kept it assertable
    # rows sorted weakest-first
    assert rep["rows"][0]["id"] == "stale"


def test_faded_helper() -> None:
    facts = [{"id": "a", "rank": 1, "daysSince": 1000, "reinforcements": 0}]
    assert faded(facts, assert_threshold=0.5) == ["a"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
