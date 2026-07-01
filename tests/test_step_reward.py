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


# --- TRL GRPO reward wiring + eval ---------------------------------------- #

def test_grpo_reward_is_trl_shaped_and_scores_per_completion() -> None:
    from provenance_bench.step_reward import make_grpo_reward

    reward_fn = make_grpo_reward(domain="physics")
    completions = [
        "STEP: 5 W | start\nSTEP: 5 J/s | watt is joule/second",  # clean -> +1
        "STEP: 5 W | start\nSTEP: 5 J | wrong unit",              # misstep -> < 0
    ]
    scores = reward_fn(["p", "p"], completions, gold=["5 W", "5 W"])
    assert scores[0] == 1.0 and scores[1] < 0.0


def test_step_reward_offline_invariants_both_domains() -> None:
    from agent import math_verifier as mv
    from provenance_bench.step_reward import offline_invariants

    ok_phys, _ = offline_invariants(domain="physics")
    assert ok_phys
    if mv.sympy_available():
        ok_math, _ = offline_invariants(domain="math")
        assert ok_math


def test_eval_rlvr_adapter_step_mock_reports_delta() -> None:
    import argparse

    from tools.eval_rlvr_adapter import run_eval_step

    args = argparse.Namespace(mode="mock", step_domain="physics", eval_frac=0.3, seed=0,
                              limit=0, adapter=None, model="mock", max_new_tokens=128)
    report = run_eval_step(args)
    assert report["task"] == "step"
    assert report["delta"]["passAt1"] > 0  # mock adapter improves verified-correct
    assert report["passed"] is True
