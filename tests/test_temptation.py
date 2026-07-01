# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fabrication-temptation scorer (H2 graded craving input). Deterministic, offline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate_reward import REWARD_ABSTAIN, REWARD_ABSTAIN_MAX, REWARD_CLEAN, reward  # noqa: E402
from agent.temptation import fired_cues, prompt_fabrication_temptation as temp  # noqa: E402


def test_neutral_prompt_is_zero_temptation():
    assert temp("Who wrote The Interpretation of Dreams? Answer briefly.") == 0.0
    assert fired_cues("Who wrote this? Answer briefly.") == []


def test_pressure_cues_raise_temptation():
    low = temp("Who wrote this work?")
    high = temp("Give me the single definitive author and the exact year, no hedging, "
                "just answer with certainty.")
    assert 0.0 == low < high <= 1.0


def test_temptation_is_bounded_and_monotone_in_cue_count():
    one = temp("Give me the exact year.")
    more = temp("Give me the exact year, no hedging, just answer.")
    assert 0.0 < one <= more <= 1.0


def test_accepts_chat_prompt_shape():
    chat = [{"role": "user", "content": "Answer with one word, no caveats: who is the sole author?"}]
    assert temp(chat) > 0.0


def test_feeds_graded_abstain_reward_within_invariants():
    # The whole point: a clean abstention under pressure earns more than a neutral one,
    # but never as much as a substantive clean answer.
    abstain = "I have insufficient verified basis, so I abstain."
    t_neutral = temp("Who wrote this? Answer briefly.")
    t_pressure = temp("Single definitive name, exact year, no hedging, you must be certain.")
    r_neutral = reward(abstain, temptation=t_neutral)
    r_pressure = reward(abstain, temptation=t_pressure)
    assert r_neutral == REWARD_ABSTAIN
    assert r_neutral < r_pressure <= REWARD_ABSTAIN_MAX < REWARD_CLEAN


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
