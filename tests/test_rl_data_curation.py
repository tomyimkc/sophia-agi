# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for provenance_bench.rl_data_curation (A4 frugal-RL primitives)."""
from __future__ import annotations

import pytest

from provenance_bench import rl_data_curation as rdc


def test_offline_invariants_pass():
    ok, detail = rdc.offline_invariants()
    assert ok, detail


def test_mixed_outcome_semantics():
    assert rdc.is_mixed_outcome([1.0, 0.0])
    assert not rdc.is_mixed_outcome([1.0, 1.0])
    assert not rdc.is_mixed_outcome([0.0, 0.0])
    assert not rdc.is_mixed_outcome([])
    assert rdc.mixed_outcome_keep([[1, 0], [1, 1]]) == [True, False]


def test_papo_asymmetry_and_ordering():
    r_out = [1.0, 0.0, 0.0]
    r_proc = [0.1, 0.9, 0.1]  # success has LOW process score; near-miss failure high
    adv = rdc.papo_advantages(r_out, r_proc, lambda_neg=0.5)
    # success stays on top regardless of its process score
    assert adv[0] > max(adv[1], adv[2])
    # near-miss failure above far failure
    assert adv[1] > adv[2]
    # lambda 0 => failures tie (outcome-only)
    adv0 = rdc.papo_advantages(r_out, r_proc, lambda_neg=0.0)
    assert adv0[1] == adv0[2]


def test_papo_group_reward_wrapper_groups_correctly():
    def outcome(prompts, completions, **kw):
        return [1.0, 0.0, 1.0, 0.0]  # two groups of 2

    def process(prompts, completions, **kw):
        return [0.5, 0.9, 0.5, 0.1]

    fn = rdc.make_papo_group_reward(outcome, process, num_generations=2, lambda_neg=0.5)
    shaped = fn(["p1", "p1", "p2", "p2"], ["a", "b", "c", "d"])
    assert len(shaped) == 4
    # within each group the success outranks the failure
    assert shaped[0] > shaped[1] and shaped[2] > shaped[3]


def test_length_mismatch_fails_closed():
    with pytest.raises(ValueError):
        rdc.papo_advantages([1.0, 0.0], [0.5])
