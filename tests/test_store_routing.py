# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Two-store routing: a knowledge signal measured on the habit store is INVALID
(quarantine), never a silent pass or a structural reject. Encodes the v4 seed-0 fix."""

from __future__ import annotations

from agent.store_routing import (
    STORE_HABIT,
    STORE_KNOWLEDGE,
    classify_signal,
    store_aware_adapter_verdict,
)


def test_signal_routing():
    assert classify_signal("learning-under-shift") == STORE_KNOWLEDGE
    assert classify_signal("distribution_shift") == STORE_KNOWLEDGE
    assert classify_signal("cpqa") == STORE_KNOWLEDGE
    assert classify_signal("eval-ladder") == STORE_HABIT
    assert classify_signal("seib") == STORE_HABIT
    assert classify_signal("religion") == STORE_HABIT
    # heuristic fallbacks
    assert classify_signal("some-new-shift-pack") == STORE_KNOWLEDGE
    assert classify_signal("mystery-metric") == STORE_HABIT


def test_habit_failure_rejects():
    v = store_aware_adapter_verdict(habit_pass=False, habit_reasons=["religion regressed"])
    assert v.verdict == "reject"
    assert any("religion" in r for r in v.reasons)


def test_v4_seed0_case_quarantines_not_rejects():
    """The actual v4 seed-0 situation: habit store passed (ladder up, retention held),
    but learning-under-shift was measured on the frozen adapter. That is invalid, not a
    reject — quarantine and re-measure on the knowledge store."""
    v = store_aware_adapter_verdict(
        habit_pass=True,
        knowledge_goals=["learning-under-shift"],
        mismatch_signals=["learning-under-shift"],
    )
    assert v.verdict == "quarantine"
    assert v.habit_verdict == "pass"
    assert any("INVALID measurement" in r for r in v.reasons)


def test_unvalidated_knowledge_goal_quarantines():
    v = store_aware_adapter_verdict(
        habit_pass=True, knowledge_goals=["distribution-shift"], knowledge_validated=False,
    )
    assert v.verdict == "quarantine"
    assert any("not yet validated" in r for r in v.reasons)


def test_clean_habit_adapter_promotes():
    v = store_aware_adapter_verdict(habit_pass=True)
    assert v.verdict == "promote"


def test_knowledge_validated_on_graph_promotes():
    v = store_aware_adapter_verdict(
        habit_pass=True, knowledge_goals=["cpqa"], knowledge_validated=True,
    )
    assert v.verdict == "promote"


def test_verdict_serializes():
    d = store_aware_adapter_verdict(habit_pass=True, knowledge_goals=["cpqa"],
                                    mismatch_signals=["learning-under-shift"]).to_dict()
    assert set(d) == {"verdict", "habitVerdict", "reasons", "knowledgeStore", "note"}
    assert d["knowledgeStore"]["route"].startswith("graph/retrieval")
