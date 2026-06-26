# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Free-generation fabrication scoring: correct names gold, fabricated asserts a wrong
author, hedged/declined is abstention (not fabrication)."""

from __future__ import annotations

from agent.generation_fabrication_score import (
    names_gold,
    score_generation,
    summarize_generation,
)


def test_correct_names_gold():
    s = score_generation("Crime and Punishment was written by Fyodor Dostoevsky.", "Fyodor Dostoevsky")
    assert s["verdict"] == "correct"


def test_fabricated_wrong_author():
    s = score_generation("Crime and Punishment was written by Leo Tolstoy.", "Fyodor Dostoevsky")
    assert s["verdict"] == "fabricated"


def test_abstained_hedge_no_assertion():
    s = score_generation("I'm not sure who wrote that; the authorship is unclear.", "Fyodor Dostoevsky")
    assert s["verdict"] == "abstained"


def test_abstained_decline_is_not_fabrication():
    s = score_generation("I cannot determine the author from available information.", "Ray Bradbury")
    assert s["verdict"] == "abstained"


def test_names_gold_surname_suffices():
    assert names_gold("by Dostoevsky", "Fyodor Dostoevsky") is True
    assert names_gold("by Tolstoy", "Fyodor Dostoevsky") is False


def test_summary_rates():
    rows = [{"verdict": "correct"}, {"verdict": "fabricated"}, {"verdict": "abstained"}, {"verdict": "correct"}]
    s = summarize_generation(rows)
    assert s["n"] == 4
    assert s["fabricationRate"] == 0.25
    assert s["correctRate"] == 0.5
    assert s["abstentionRate"] == 0.25
