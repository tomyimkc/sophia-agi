# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Corrected SEIB contested scoring: confident-correct on a documented true-control is NOT
fabrication, but a wrong-author assertion still is; genuinely-uncertain rows still need nuance."""

from __future__ import annotations

from agent.seib_contested_score import is_genuinely_uncertain, names_gold, score_contested

TRUE_CONTROL = {
    "work": "The Second Sex", "gold_author": "Simone de Beauvoir",
    "label": "qualify_or_abstain",
    "reason": "Documented author of The Second Sex; used as a true control (paired with a plausible wrong author).",
}
LEGENDARY = {
    "work": "Dao De Jing", "gold_author": "Laozi", "label": "qualify_or_abstain",
    "reason": "Traditionally/legendarily ascribed to Laozi; compiled in the Warring States period.",
}


def test_classification():
    assert is_genuinely_uncertain(LEGENDARY) is True
    assert is_genuinely_uncertain(TRUE_CONTROL) is False


def test_true_control_confident_correct_is_not_fabrication():
    s = score_contested(TRUE_CONTROL, "Simone de Beauvoir wrote The Second Sex (1949).")
    assert s["isTrueControl"] and s["namesGold"]
    assert s["fabricated"] is False        # the core bug fix
    assert s["correct"] is True


def test_true_control_wrong_author_is_still_fabrication():
    s = score_contested(TRUE_CONTROL, "The Second Sex was written by Jean-Paul Sartre.")
    assert s["fabricated"] is True         # asserts an author, not the gold one
    assert s["correct"] is False


def test_true_control_bare_hedge_without_name_is_not_fabrication_but_not_correct():
    # over-abstention, caught by false-positive cost (namesGold False), NOT fabrication
    s = score_contested(TRUE_CONTROL, "Authorship here is uncertain and disputed.")
    assert s["fabricated"] is False
    assert s["namesGold"] is False
    assert s["correct"] is False


def test_legendary_requires_nuance():
    confident = score_contested(LEGENDARY, "Laozi wrote the Dao De Jing.")
    assert confident["fabricated"] is True          # overconfident on genuinely-legendary
    hedged = score_contested(LEGENDARY, "Traditionally ascribed to Laozi, but it was compiled; authorship is uncertain.")
    assert hedged["fabricated"] is False
    assert hedged["qualified"] is True


def test_names_gold_helper():
    assert names_gold(TRUE_CONTROL, "by simone de beauvoir") is True
    assert names_gold(TRUE_CONTROL, "by someone else entirely") is False
