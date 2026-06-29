# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""H2 graded-craving invariants for agent.gate_reward (Atomic Habits, make-it-attractive).

Guards the abstention-collapse fix: graded craving may make the *hardest* honest refusal
more attractive, but a refusal must stay strictly reward-positive, strictly below a clean
answer, monotone in temptation, and the flat default (temptation=None) must be unchanged.
"""

from agent import gate_reward as gr

ABSTAIN = (
    "I have insufficient verified basis to attribute this work to any individual, "
    "so I will abstain."
)
CLEAN = "The Project Phoenix Charter was written by the founding committee."


def test_flat_default_unchanged():
    # The whole point: existing callers (temptation omitted/None) see no behaviour change.
    assert gr.reward(ABSTAIN) == gr.REWARD_ABSTAIN
    assert gr.reward(ABSTAIN, temptation=None) == gr.REWARD_ABSTAIN


def test_graded_endpoints_and_monotonicity():
    assert gr.graded_abstain_reward(0.0) == gr.REWARD_ABSTAIN
    assert gr.graded_abstain_reward(1.0) == gr.REWARD_ABSTAIN_MAX
    seq = [gr.graded_abstain_reward(t / 10) for t in range(11)]
    assert seq == sorted(seq), "graded craving must be monotone non-decreasing in temptation"


def test_graded_stays_positive_and_below_clean():
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        r = gr.reward(ABSTAIN, temptation=t)
        assert 0.0 < r <= gr.REWARD_ABSTAIN_MAX < gr.REWARD_CLEAN
        assert gr.REWARD_VIOLATION < r  # abstention always beats a violation


def test_temptation_is_clamped():
    assert gr.reward(ABSTAIN, temptation=-5.0) == gr.REWARD_ABSTAIN
    assert gr.reward(ABSTAIN, temptation=5.0) == gr.REWARD_ABSTAIN_MAX


def test_clean_answer_ignores_temptation():
    # Graded craving only touches the abstention branch; a substantive clean answer is
    # always REWARD_CLEAN regardless of temptation.
    assert gr.reward(CLEAN, temptation=1.0) == gr.REWARD_CLEAN
    assert gr.reward(CLEAN, temptation=0.0) == gr.REWARD_CLEAN


def test_grpo_reward_with_temptation_fn():
    # A graded reward_fn pairs each completion with its prompt's temptation.
    fn = gr.make_grpo_reward(temptation_fn=lambda prompt, comp: 1.0)
    out = fn(["p1", "p2"], [ABSTAIN, CLEAN])
    assert out == [gr.REWARD_ABSTAIN_MAX, gr.REWARD_CLEAN]
    assert fn.__name__ == "sophia_gate_reward_graded"
    # Without a temptation_fn it is exactly the flat reward.
    flat = gr.make_grpo_reward()
    assert flat(["p"], [ABSTAIN]) == [gr.REWARD_ABSTAIN]
    assert flat.__name__ == "sophia_gate_reward"


def test_self_check_passes_graded_invariants():
    detail = gr.self_check()
    inv = detail["invariants"]
    assert inv["gradedMonotone"] and inv["gradedPositiveBelowClean"]
    assert inv["flatDefaultUnchanged"] and inv["bounded"]
