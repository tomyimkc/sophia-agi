# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for provenance_bench/step_reward.py and tools/run_step_rlvr.py."""
from __future__ import annotations

from agent import math_verifier as mv
from provenance_bench.step_reward import reward_for_derivation
from tools.run_step_rlvr import check_invariants


def test_clean_physics_chain_rewards_plus_one() -> None:
    score, detail = reward_for_derivation([{"expr": "5 W"}, {"expr": "5 J/s"}], gold="5 W", domain="physics")
    assert score == 1.0
    assert detail["verdict"] == "accepted"


def test_dimension_error_is_penalised() -> None:
    score, detail = reward_for_derivation([{"expr": "5 W"}, {"expr": "5 J"}], gold="5 W", domain="physics")
    assert score < 0.0
    assert detail["verdict"] == "rejected"


def test_bounded_and_deterministic() -> None:
    a, _ = reward_for_derivation([{"expr": "1 km"}, {"expr": "1000 m"}], domain="physics")
    b, _ = reward_for_derivation([{"expr": "1 km"}, {"expr": "1000 m"}], domain="physics")
    assert a == b and -1.0 <= a <= 1.0


def test_unverifiable_scores_zero() -> None:
    score, detail = reward_for_derivation([{"expr": "5 W"}], domain="physics")
    assert score == 0.0
    assert detail["nChecks"] == 0


def test_misstep_cannot_outvote_to_positive() -> None:
    if not mv.sympy_available():
        return
    # Many correct steps then one wrong final step must NOT net out positive.
    steps = [{"expr": "(x+1)**2"}, {"expr": "x**2 + 2*x + 1"}, {"expr": "x**2 + 2*x + 1"}, {"expr": "x**2 + 9"}]
    score, _ = reward_for_derivation(steps, gold="x**2 + 2*x + 1", domain="math")
    assert score <= 0.0


def test_offline_invariants_pass() -> None:
    assert check_invariants() == []
