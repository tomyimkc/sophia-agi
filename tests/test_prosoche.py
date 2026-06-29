# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prosoche attention gate — routing fidelity + reward-axis invariants (offline)."""
from __future__ import annotations

from agent.prosoche import (
    DRIFT_WEIGHTS,
    FOCUS_DRIFT,
    FOCUS_FIXATION,
    FOCUS_ONGOAL,
    FOCUS_REANCHOR,
    AttentionAnchor,
    anchor_segment,
    assess_attention,
    focus_reward_axis,
    prosoche_quotient,
    self_check,
)

ANCHOR = AttentionAnchor(
    goal="fix the failing auth login test in services.auth",
    in_scope_axes=("provenance",),
    in_scope_entities=("services.auth", "login", "auth test"),
)


def test_self_check_invariants():
    rep = self_check()
    assert rep["driftWeightsSumToOne"]
    assert rep["rewardOrder"] == [FOCUS_ONGOAL, FOCUS_REANCHOR, FOCUS_DRIFT, FOCUS_FIXATION]


def test_on_goal_is_focused():
    d = assess_attention(
        "Looking at the login test in services.auth: the auth token check rejects valid sessions.",
        ANCHOR,
    )
    assert d.verdict == "focused"
    assert d.pq >= 0.55


def test_off_goal_is_drifting():
    d = assess_attention(
        "While I'm here, let me rewrite the unrelated Marketing Page and recolour the Telemetry Dashboard.",
        ANCHOR,
    )
    assert d.verdict == "drifting"


def test_legitimate_shift_redirected_is_reanchor():
    d = assess_attention(
        "The user changed the goal — re-anchoring: the new objective is the logout flow.",
        ANCHOR,
        context={"goalShift": True},
    )
    assert d.verdict == "re-anchor"
    assert d.goalShift


def test_legitimate_shift_ignored_escalates():
    d = assess_attention(
        "Ignoring that; I'll keep tuning the original login assertion as planned.",
        ANCHOR,
        context={"goalShift": True},
    )
    assert d.verdict == "escalate"


def test_decline_distractor_stays_focused():
    d = assess_attention(
        "That dashboard refactor is out of scope for the current goal; back to the login test in services.auth.",
        ANCHOR,
    )
    assert d.verdict == "focused"


def test_pq_monotonic_on_goal_vs_off_goal():
    on = prosoche_quotient("the login auth test in services.auth fails", ANCHOR)
    off = prosoche_quotient("quarterly sales revenue and marketing spend forecast", ANCHOR)
    assert on > off


def test_anchor_is_pinned_and_stable():
    seg = anchor_segment(ANCHOR)
    assert seg.pinned and seg.stable and not seg.compressible
    assert ANCHOR.id in seg.provenance


def test_anchor_id_changes_with_goal():
    a2 = AttentionAnchor(goal="a different goal entirely", in_scope_entities=ANCHOR.in_scope_entities)
    assert a2.id != ANCHOR.id


def test_anchor_from_dict_camel_and_snake():
    a = AttentionAnchor.from_dict({"goal": "g", "inScopeEntities": ["x"], "inScopeAxes": ["provenance"]})
    assert a.goal == "g" and a.in_scope_entities == ("x",) and a.in_scope_axes == ("provenance",)


def test_reward_ordering_strict():
    on = focus_reward_axis("login auth test in services.auth is failing", ANCHOR)
    re_ = focus_reward_axis("re-anchoring: the new goal is the logout flow", ANCHOR, goal_shift=True)
    dr = focus_reward_axis("let me rewrite the unrelated marketing page now", ANCHOR)
    fx = focus_reward_axis("ignoring that, continuing the original plan", ANCHOR, goal_shift=True)
    assert on >= re_ > 0 > dr > fx


def test_empty_goal_does_not_crash():
    a = AttentionAnchor(goal="")
    d = assess_attention("anything at all", a)
    assert d.verdict in ("focused", "drifting", "escalate", "re-anchor")
