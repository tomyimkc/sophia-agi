# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""H4 habit strength (Atomic Habits: make-it-satisfying / never-miss-twice).

A behaviour consolidates with reinforcement (spaced-repetition via the forgetting curve);
a single slip costs one stability step (not a reset); two slips in a row trip an at-risk
tripwire. Deterministic, offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import forgetting_curve as fc  # noqa: E402
from agent.habit_strength import NEVER_MISS_TWICE, HabitStrength  # noqa: E402


def test_reinforcement_consolidates_strength():
    h = HabitStrength()
    weak = h.strength("b", days_since=30)            # never practised
    for _ in range(5):
        h.reinforce("b")
    assert h.strength("b", days_since=30) > weak
    # monotone in reinforcement count at a fixed disuse interval
    seq = []
    g = HabitStrength()
    for _ in range(6):
        seq.append(g.strength("b", 30))
        g.reinforce("b")
    assert seq == sorted(seq)


def test_one_miss_is_one_step_not_a_reset():
    h = HabitStrength()
    for _ in range(5):
        h.reinforce("b")
    h.miss("b")
    assert h.reinforcements("b") == 4
    # strength equals a clean 4-reinforcement habit (one step back), not zero
    ref = HabitStrength()
    for _ in range(4):
        ref.reinforce("b")
    assert abs(h.strength("b", 30) - ref.strength("b", 30)) < 1e-9
    assert h.strength("b", 30) > 0.0


def test_never_miss_twice_tripwire_and_recovery():
    h = HabitStrength()
    for _ in range(3):
        h.reinforce("b")
    h.miss("b")
    assert not h.at_risk("b")                 # one miss = accident
    h.miss("b")
    assert h.at_risk("b") and NEVER_MISS_TWICE == 2
    h.reinforce("b")                          # recovery clears the streak
    assert not h.at_risk("b")


def test_miss_floors_reinforcements_at_zero():
    h = HabitStrength()
    h.miss("b")
    assert h.reinforcements("b") == 0
    assert 0.0 <= h.strength("b", 0) <= 1.0


def test_report_flags_faded_and_at_risk():
    h = HabitStrength(assert_threshold=0.5)
    for _ in range(5):
        h.reinforce("strong")
    h.reinforce("weak")                       # barely practised
    h.miss("risky"); h.miss("risky")          # missed twice
    rep = h.report({"strong": 10, "weak": 120, "risky": 0})
    assert rep["candidateOnly"] is True
    assert "weak" in rep["faded"]             # long disuse + little practice => faded
    assert "strong" not in rep["faded"]
    assert "risky" in rep["atRisk"]
    # rows sorted ascending by strength
    strengths = [r["strength"] for r in rep["rows"]]
    assert strengths == sorted(strengths)


def test_shares_forgetting_curve_math():
    # Strength must equal forgetting_curve.retention of the net reinforcement count.
    h = HabitStrength()
    for _ in range(3):
        h.reinforce("b")
    assert abs(h.strength("b", 45) - fc.retention(45, 3)) < 1e-12


def test_self_check_passes():
    from agent.habit_strength import self_check
    inv = self_check()["invariants"]
    assert all(inv.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
