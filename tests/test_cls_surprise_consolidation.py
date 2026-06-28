# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Integration test: surprise-gated consolidation cooperates with stability selection.

The thesis-level property: surprise PROPOSES, stability DISPOSES, and the anti-forgetting
gate still has the final say. This test only covers the selection layer; the plasticity
gate is exercised by its own suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cls_consolidation import select_consolidation_set  # noqa: E402
from okf.surprise_consolidation import surprise_gated_consolidation_set  # noqa: E402


def test_stability_only_baseline_unchanged():
    """Without a decay plan, surprise-gated selection == stability selection (no regression).

    The stability baseline comes from the live ``agent.cls_consolidation`` path; the
    surprise augmentation is the candidate-only selector in ``okf.surprise_consolidation``.
    When no decay plan is supplied they must agree exactly.
    """
    streaks = {"stable_fact": 5, "new_fact": 1, "unstable": 0}
    cleared = {"stable_fact", "new_fact", "unstable"}
    base = select_consolidation_set(streaks, cleared, min_stable_snapshots=3)
    aug = surprise_gated_consolidation_set(streaks, cleared, decay_plan=None, min_stable_snapshots=3)
    assert base == aug == ["stable_fact"]


def test_surprise_proposes_stability_disposes_default():
    """A surprise-reinforced fact that is NOT yet stable is held back by default."""
    streaks = {"novel": 1, "stable": 5}            # novel not stable yet
    cleared = {"novel", "stable"}
    plan = {"reinforce": ["novel"]}                 # decay says: consolidate novel
    aug = surprise_gated_consolidation_set(streaks, cleared, plan, min_stable_snapshots=3)
    # default: novel must also be stable -> held back (stability disposes)
    assert aug == ["stable"]


def test_surprise_can_consolidate_ahead_of_stability_when_enabled():
    """A high-surprise reinforced belief may consolidate immediately for frontier problems."""
    streaks = {"novel": 1, "stable": 5}
    cleared = {"novel", "stable"}
    plan = {"reinforce": ["novel"]}
    aug = surprise_gated_consolidation_set(streaks, cleared, plan,
                                           min_stable_snapshots=3, include_surprise_only=True)
    assert "novel" in aug and "stable" in aug


def test_surprise_belief_must_be_gate_cleared():
    """Surprise alone never bypasses the gate — a surprised belief not in gate_cleared is excluded."""
    streaks = {"novel": 5}
    cleared = set()                                  # NOT gate-cleared
    plan = {"reinforce": ["novel"]}
    aug = surprise_gated_consolidation_set(streaks, cleared, plan, include_surprise_only=True)
    assert aug == []


def test_two_signals_union_dedup():
    """Stable + surprise sets union and dedup cleanly."""
    streaks = {"a": 5, "b": 5, "c": 1}
    cleared = {"a", "b", "c"}
    plan = {"reinforce": ["b", "c"]}                 # b is both stable+surprise; c surprise-only
    aug = surprise_gated_consolidation_set(streaks, cleared, plan, include_surprise_only=True)
    assert aug == ["a", "b", "c"]                    # deduped, sorted


def main() -> int:
    test_stability_only_baseline_unchanged()
    test_surprise_proposes_stability_disposes_default()
    test_surprise_can_consolidate_ahead_of_stability_when_enabled()
    test_surprise_belief_must_be_gate_cleared()
    test_two_signals_union_dedup()
    print("test_cls_surprise_consolidation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
