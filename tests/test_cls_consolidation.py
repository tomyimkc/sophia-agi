#!/usr/bin/env python3
"""Tests for agent.cls_consolidation — consolidate stable wiki facts into weights,
only through the anti-forgetting plasticity gate.

Verifies that only facts stable for >= N snapshots AND gate-cleared are selected, that
a clean improving adapter is promoted, and the central invariant: an adapter that
regresses a protected suite (old knowledge) is REJECTED — weights can never gain new
knowledge at the cost of forgetting old. Offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cls_consolidation import consolidate, select_consolidation_set, stability_streaks  # noqa: E402
from agent.continual_retention import Snapshot  # noqa: E402


def _snapshots() -> "list[Snapshot]":
    # 'stable' grounded at strength 3 across all 4 steps; 'late' appears only at the end;
    # 'wobbly' drops confidence midway (breaks its streak).
    return [
        Snapshot("t1", {"stable": 3, "wobbly": 3}, ("stable", "wobbly")),
        Snapshot("t2", {"stable": 3, "wobbly": 3}, ()),
        Snapshot("t3", {"stable": 3, "wobbly": 1}, ()),
        Snapshot("t4", {"stable": 3, "wobbly": 1, "late": 4}, ("late",)),
    ]


def test_stability_streaks_count_trailing_stable() -> None:
    streaks = stability_streaks(_snapshots())
    assert streaks["stable"] == 4          # grounded & strong every step
    assert streaks["wobbly"] == 0          # confidence fell below origin -> streak broken
    assert streaks["late"] == 1            # only present in the final snapshot


def test_selection_requires_stability_and_gate_clearance() -> None:
    streaks = stability_streaks(_snapshots())
    # 'stable' qualifies on streak; require it also be gate-cleared.
    assert select_consolidation_set(streaks, gate_cleared=["stable"], min_stable_snapshots=3) == ["stable"]
    # Stable but not gate-cleared -> excluded (fail-closed).
    assert select_consolidation_set(streaks, gate_cleared=[], min_stable_snapshots=3) == []
    # 'late' is gate-cleared but not stable enough -> excluded.
    assert select_consolidation_set(streaks, gate_cleared=["late"], min_stable_snapshots=3) == []


def test_clean_improving_adapter_promotes() -> None:
    report = consolidate(
        _snapshots(), gate_cleared=["stable"],
        metrics=[("tool_routing", 0.72, 0.80), ("source_discipline", 0.98, 0.98),
                 ("fact_check_false_accept", 0.99, 0.99)],
        target_suite="tool_routing", verifier_artifacts=("heldout", "provenance-delta"),
    )
    assert report["selected"] == ["stable"]
    assert report["decision"]["verdict"] == "promote"
    assert report["antiForgettingEnforced"] is True


def test_adapter_that_forgets_protected_knowledge_is_rejected() -> None:
    # New skill improves, but source_discipline regresses -> catastrophic forgetting.
    report = consolidate(
        _snapshots(), gate_cleared=["stable"],
        metrics=[("tool_routing", 0.72, 0.95), ("source_discipline", 0.98, 0.90),
                 ("fact_check_false_accept", 0.99, 0.99)],
        target_suite="tool_routing", verifier_artifacts=("heldout", "provenance-delta"),
    )
    assert report["decision"]["verdict"] == "reject"      # the anti-forgetting tripwire
    assert report["antiForgettingEnforced"] is True


def test_nothing_stable_yet_builds_no_candidate() -> None:
    # Require a longer streak than any fact has -> consolidation waits.
    report = consolidate(_snapshots(), gate_cleared=["stable"], metrics=[],
                         target_suite="tool_routing", min_stable_snapshots=99)
    assert report["selected"] == []
    assert report["decision"] is None
    assert report["antiForgettingEnforced"] is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
