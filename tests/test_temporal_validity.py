#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.temporal_validity — continual learning over a changing truth.

Verifies as-of-date belief: a fact whose window closed is gone, its successor appears,
timeless facts always hold, and overlapping supersession windows are flagged. Offline,
deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.temporal_validity import belief_state_as_of, temporal_conflicts, valid_at  # noqa: E402
from okf.page import Page  # noqa: E402


def _p(pid, **meta):
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept",
                "authorConfidence": "consensus", **meta})


def _timeline():
    return [
        _p("pluto_planet", validUntil="2006", supersededBy=["pluto_dwarf"]),
        _p("pluto_dwarf", validFrom="2006", supersedes=["pluto_planet"]),
        _p("water_boils_100c"),   # timeless
    ]


def test_valid_at_window() -> None:
    assert valid_at({"validUntil": "2006"}, "2000") is True
    assert valid_at({"validUntil": "2006"}, "2010") is False
    assert valid_at({"validFrom": "2006"}, "2000") is False
    assert valid_at({}, "1999") is True            # timeless


def test_belief_state_as_of_switches_at_boundary() -> None:
    before = belief_state_as_of(_timeline(), "2000")
    after = belief_state_as_of(_timeline(), "2010")
    assert "pluto_planet" in before and "pluto_dwarf" not in before
    assert "pluto_dwarf" in after and "pluto_planet" not in after
    assert "water_boils_100c" in before and "water_boils_100c" in after   # timeless always


def test_temporal_conflicts_flags_overlap() -> None:
    # Overlap: older fact has no validUntil but a newer one supersedes it from 2006.
    overlapping = [_p("a", supersedes=["b"], validFrom="2006"), _p("b")]
    assert temporal_conflicts(overlapping)
    # Clean: older ends exactly when newer begins.
    clean = [_p("a2", supersedes=["b2"], validFrom="2006"), _p("b2", validUntil="2006")]
    assert temporal_conflicts(clean) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
