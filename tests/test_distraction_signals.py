# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Distraction / fixation dual-signals (offline, informational-only)."""
from __future__ import annotations

from agent.distraction_signals import detect_distraction, self_check
from agent.prosoche import AttentionAnchor

ANCHOR = AttentionAnchor(
    goal="fix the failing auth login test in services.auth",
    in_scope_entities=("services.auth", "login", "auth test"),
)


def test_self_check():
    rep = self_check()
    assert rep["distraction"]["axis"] == "distraction"
    assert rep["fixation"]["axis"] == "fixation"
    assert rep["safetyFixation"]["risk"] == 1.0


def test_clean_step_no_vice():
    rep = detect_distraction(
        "The login test in services.auth fails: the auth token check rejects valid sessions.",
        ANCHOR,
    )
    assert rep.axis == "none"
    assert rep.risk == 0.0


def test_distraction_detected():
    rep = detect_distraction(
        "While I'm here, let me also refactor the unrelated Billing Service and the Email Templates.",
        ANCHOR,
    )
    assert rep.axis == "distraction"
    assert rep.risk > 0


def test_fixation_on_declared_shift():
    rep = detect_distraction(
        "Continuing with the original login assertion regardless.",
        ANCHOR,
        context={"goalShift": True},
    )
    assert rep.axis == "fixation"


def test_fixation_is_informational_only():
    # The report exposes signals but cannot itself force an action — it is a
    # dataclass of advice, not a verdict that suppresses output.
    rep = detect_distraction("anything", ANCHOR)
    assert hasattr(rep, "axis") and hasattr(rep, "risk") and hasattr(rep, "reasons")
    assert not hasattr(rep, "block") and not hasattr(rep, "suppress")
