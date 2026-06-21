#!/usr/bin/env python3
"""Tests for the long-horizon autonomy curve (agent/horizon.py). Offline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import horizon as h  # noqa: E402


def test_chain_task_deterministic_and_correct() -> None:
    t1 = h.make_chain_task(5, seed=42)
    t2 = h.make_chain_task(5, seed=42)
    assert t1 == t2 and t1["length"] == 5 and isinstance(t1["gold"], int)
    # the gold is what a perfect solver returns
    assert h._final_int(h.perfect_solver(t1)) == t1["gold"]


def test_perfect_solver_full_horizon() -> None:
    res = h.horizon_curve(h.perfect_solver, lengths=(1, 4, 16), trials=10)
    assert all(c["successRate"] == 1.0 for c in res["curve"])
    assert res["effectiveHorizon"] == 16


def test_noisy_solver_decays_with_length() -> None:
    res = h.horizon_curve(h.noisy_solver(0.5, seed=1), lengths=(1, 16), trials=40)
    rates = {c["length"]: c["successRate"] for c in res["curve"]}
    assert rates[1] > rates[16]                 # longer tasks are harder
    assert res["effectiveHorizon"] < 16         # high per-step error -> short horizon


def test_final_int_parsing() -> None:
    assert h._final_int("the answer is 42") == 42
    assert h._final_int("first 3 then 7") == 7
    assert h._final_int("no number") is None


def main() -> int:
    test_chain_task_deterministic_and_correct()
    test_perfect_solver_full_horizon()
    test_noisy_solver_decays_with_length()
    test_final_int_parsing()
    print("test_horizon: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
