#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.knowledge_gap_log — the self-improving corpus loop.

Verifies clean grounded answers are NOT logged, gap policies are, and the worklist ranks
by query frequency while seeding from audit thin targets. Writes to a temp path; offline,
deterministic, dependency-free.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.knowledge_gap_log import gap_worklist, is_gap, load_gaps, log_gap  # noqa: E402


def test_only_gap_policies_logged() -> None:
    assert is_gap("abstain_no_route") and is_gap("grounded_fallback")
    assert not is_gap("grounded_strict")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gaps.jsonl"
        assert log_gap("q1", target="x", policy="grounded_strict", path=path) is None   # clean -> not logged
        assert log_gap("q2", target="y", policy="abstain_no_route", path=path) is not None
        assert len(load_gaps(path)) == 1


def test_worklist_ranks_by_frequency_and_seeds_thin() -> None:
    gaps = [
        {"target": "y", "policy": "grounded_fallback"},
        {"target": "y", "policy": "grounded_fallback"},
        {"target": "z", "policy": "abstain_no_source"},
        {"target": None, "policy": "abstain_no_route"},
    ]
    wl = gap_worklist(gaps, thin_targets=["w"])
    items = wl["worklist"]
    assert items[0]["target"] == "y" and items[0]["gapHits"] == 2     # most-queried first
    assert wl["unroutableQueries"] == 1
    # 'w' seeded from audit despite zero query hits
    assert any(i["target"] == "w" and i["auditThin"] and i["gapHits"] == 0 for i in items)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
