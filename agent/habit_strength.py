# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Habit strength for behaviours (H4) — spaced-repetition consolidation + "never miss twice".

Design note: docs/06-Roadmap/Atomic-Habits-for-Sophia.md.

``agent.forgetting_curve`` already gives *facts* a spaced-repetition curve: each reinforcement
raises stability, so used facts decay slowly and unused ones fade gracefully. Atomic Habits
says a *habit* consolidates the same way — repetition raises its strength — and adds one rule
this module makes literal: **never miss twice**. Missing once is an accident; missing twice is
the start of a new (bad) habit. So a single slip costs *one* stability step (not a reset), and
the tracker raises an at-risk tripwire the moment misses occur back-to-back.

This generalises the forgetting curve from facts to *behaviours* (a verifier firing, a skill
being exercised, the gate abstaining when it should). Strength is the forgetting-curve retention
of the behaviour's net reinforcement count after a given disuse interval — so a long-practised
habit survives a lapse and a gap, while an unpractised one fades below the assertion threshold
and is flagged for re-practice.

Pure, deterministic, offline — elapsed time is passed in (no wall-clock), reusing
``agent.forgetting_curve`` so the math stays a single source of truth.

    h = HabitStrength()
    for _ in range(5): h.reinforce("abstain-on-trap")   # practise the habit
    h.strength("abstain-on-trap", days_since=30)         # high — consolidated
    h.miss("abstain-on-trap")                            # one slip: one step back, NOT zero
    h.at_risk("abstain-on-trap")                         # False — a single miss is an accident
    h.miss("abstain-on-trap"); h.at_risk("abstain-on-trap")  # True — missed twice
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import forgetting_curve as fc

# Missing twice in a row is the book's danger line ("the start of a new habit").
NEVER_MISS_TWICE = 2


@dataclass
class _State:
    reinforcements: int = 0       # net practised count (raises stability)
    misses: int = 0               # total slips (audit only)
    consecutive_misses: int = 0   # resets to 0 on any reinforcement


class HabitStrength:
    """Track per-behaviour habit strength via the forgetting curve + never-miss-twice.

    ``base_days`` / ``per_reinforcement`` are forwarded to ``forgetting_curve.stability`` so
    this tracker and the belief-graph forgetting curve share one decay model. ``assert_threshold``
    is the retention below which a behaviour is considered *faded* (needs re-practice).
    """

    def __init__(self, *, base_days: float = 30.0, per_reinforcement: float = 30.0,
                 assert_threshold: float = 0.5) -> None:
        self.base_days = base_days
        self.per_reinforcement = per_reinforcement
        self.assert_threshold = assert_threshold
        self._behaviours: "dict[str, _State]" = {}

    def _st(self, behaviour: str) -> _State:
        return self._behaviours.setdefault(behaviour, _State())

    def reinforce(self, behaviour: str) -> None:
        """A correct exercise of the habit: +1 stability step, clears the consecutive-miss streak."""
        s = self._st(behaviour)
        s.reinforcements += 1
        s.consecutive_misses = 0

    def miss(self, behaviour: str) -> None:
        """A slip. Never-miss-twice: one miss costs exactly ONE stability step (floored at 0),
        not a reset — so a long-practised habit survives an accident. Back-to-back misses each
        cost a step and trip the at-risk tripwire."""
        s = self._st(behaviour)
        s.reinforcements = max(0, s.reinforcements - 1)
        s.misses += 1
        s.consecutive_misses += 1

    def reinforcements(self, behaviour: str) -> int:
        return self._st(behaviour).reinforcements

    def strength(self, behaviour: str, days_since: float = 0.0) -> float:
        """Habit strength in [0, 1] — forgetting-curve retention of the net reinforcement count
        after ``days_since`` of disuse. Higher reinforcement => slower decay."""
        s = self._st(behaviour)
        return fc.retention(days_since, s.reinforcements,
                            base_days=self.base_days, per_reinforcement=self.per_reinforcement)

    def at_risk(self, behaviour: str) -> bool:
        """True once the behaviour has been missed ``NEVER_MISS_TWICE`` times in a row — the
        moment a lapse stops being an accident and becomes an emerging anti-habit."""
        return self._st(behaviour).consecutive_misses >= NEVER_MISS_TWICE

    def report(self, behaviour_days: "dict[str, float]") -> "dict[str, Any]":
        """Strength report over behaviours. ``behaviour_days`` maps behaviour -> days_since_last_use.
        Flags faded behaviours (strength < assert_threshold) and at-risk ones (missed twice)."""
        rows = []
        faded, at_risk = [], []
        for b, days in behaviour_days.items():
            s = self._st(b)
            strength = round(self.strength(b, days), 4)
            is_faded = strength < self.assert_threshold
            is_risk = self.at_risk(b)
            rows.append({
                "behaviour": b, "reinforcements": s.reinforcements, "misses": s.misses,
                "consecutiveMisses": s.consecutive_misses, "daysSince": days,
                "strength": strength, "faded": is_faded, "atRisk": is_risk,
            })
            if is_faded:
                faded.append(b)
            if is_risk:
                at_risk.append(b)
        rows.sort(key=lambda r: r["strength"])
        return {
            "schema": "sophia.habit_strength.v1",
            "candidateOnly": True,
            "assertThreshold": self.assert_threshold,
            "total": len(rows),
            "fadedCount": len(faded),
            "faded": faded,
            "atRiskCount": len(at_risk),
            "atRisk": at_risk,
            "rows": rows,
        }


def self_check() -> dict:
    """Offline assertion of the H4 invariants (no wall-clock / GPU)."""
    h = HabitStrength()
    b = "abstain-on-trap"
    for _ in range(5):
        h.reinforce(b)
    s5 = h.strength(b, days_since=30)

    # Reinforcement consolidates: more practice => higher strength at the same disuse interval.
    h2 = HabitStrength()
    h2.reinforce(b)
    assert h2.strength(b, 30) < s5, "more reinforcement must raise strength"

    # Never-miss-twice: a single miss is exactly one stability step back (== 4 reinforcements),
    # NOT a reset to zero.
    h.miss(b)
    ref = HabitStrength()
    for _ in range(4):
        ref.reinforce(b)
    assert abs(h.strength(b, 30) - ref.strength(b, 30)) < 1e-9, "one miss must be one step, not a reset"
    assert h.reinforcements(b) == 4, "single miss decrements by one"
    assert not h.at_risk(b), "a single miss is an accident, not at-risk"

    # Missing twice in a row trips the tripwire.
    h.miss(b)
    assert h.at_risk(b), "missed twice => at risk"
    # A reinforcement clears the streak (recovery).
    h.reinforce(b)
    assert not h.at_risk(b), "reinforcement clears the consecutive-miss streak"

    # Strength is bounded and floors gracefully (never negative, misses below 0 reinforcements ok).
    empty = HabitStrength()
    empty.miss("never-practised")
    assert 0.0 <= empty.strength("never-practised", 0) <= 1.0
    assert empty.reinforcements("never-practised") == 0, "miss floors reinforcements at 0"

    return {
        "candidateOnly": True,
        "strengthAfter5x30d": round(s5, 4),
        "invariants": {
            "reinforcementConsolidates": h2.strength(b, 30) < s5,
            "oneMissOneStep": True,
            "missedTwiceTripsRisk": True,
            "reinforcementRecovers": True,
            "boundedAndFloored": True,
        },
    }


if __name__ == "__main__":
    detail = self_check()
    print(detail, flush=True)
    print("HABIT-STRENGTH SELF-CHECK PASSED ✓", flush=True)
