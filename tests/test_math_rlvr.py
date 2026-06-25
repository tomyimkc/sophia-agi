# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the math RLVR pack (no torch, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import math_dataset, math_reward  # noqa: E402


def test_family_disjoint_split() -> None:
    data = math_dataset.build_math_rl_dataset(eval_frac=0.34, seed=0)
    assert data["family_intersection"] == []          # contamination-free
    assert data["train_rows"] and data["eval_rows"]
    train_f = {p["family"] for p in data["train_problems"]}
    eval_f = {p["family"] for p in data["eval_problems"]}
    assert train_f.isdisjoint(eval_f)
    # rows carry the reward-routing column
    assert all("gold" in r for r in data["train_rows"])


def test_reward_failclosed_or_correct() -> None:
    score, detail = math_reward.reward_for_problem(
        r"the answer is \boxed{(x-1)(x+1)}", "(x-1)*(x+1)")
    if math_reward.sympy_available():
        assert score == math_reward.REWARD_MAX and detail["passed"]
        bad, _ = math_reward.reward_for_problem(r"\boxed{x**2+1}", "(x-1)*(x+1)")
        assert bad == math_reward.REWARD_MIN
    else:
        # fail-closed: cannot verify without sympy
        assert score == math_reward.REWARD_MIN and "sympy_unavailable" in detail["reason"]


def test_grpo_reward_shape() -> None:
    rf = math_reward.make_grpo_reward()
    out = rf(["p1", "p2"], [r"\boxed{x**2-1}", r"\boxed{nonsense+1}"],
             gold=["x**2 - 1", "x**2 - 1"])
    assert isinstance(out, list) and len(out) == 2
    assert all(math_reward.REWARD_MIN <= r <= math_reward.REWARD_MAX for r in out)


def test_offline_invariants() -> None:
    ok, detail = math_reward.offline_invariants()
    assert ok, detail["checks"]
    assert detail["familyIntersection"] == []


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("math RLVR offline invariants PASS")
