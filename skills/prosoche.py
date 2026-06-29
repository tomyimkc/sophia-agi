# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: focus_advocate — Prosoche attention gate, fail-closed.

The allocation-axis sibling of ``measure_advocate``/``courage_advocate``: where
Sophrosyne decides *how much* and Andreia decides *direction*, this decides *on
what* — whether the agent's effort is focused on the active goal, drifting onto an
out-of-scope target, hitting a legitimate goal change (re-anchor), or must escalate.
It NEVER prunes a safety-relevant step as off-goal (attention is not blindness), and
on any error it returns the conservative ``focused``.
"""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "focus_advocate",
    summary="Assess whether a step is focused|drifting|re-anchor|escalate against an attention anchor (Prosoche attention gate). Flags distraction (off-goal wandering) and fixation (ignoring a legitimate/safety shift); never prunes a safety step as off-goal.",
    uses=("attention_assess_tool", "distraction_check_tool"),
)
def focus_advocate(*, text: str, anchor: dict | None = None, context: dict | None = None) -> dict:
    d = call("attention_assess_tool", text=text, anchor=anchor, context=context)
    return {
        "verdict": d.get("verdict", "focused"),
        "pq": d.get("pq"),
        "reason": d.get("reason"),
        "anchorId": d.get("anchorId"),
        "safetyRelevant": d.get("safetyRelevant", False),
        "goalShift": d.get("goalShift", False),
        "signals": d.get("signals", {}),
        "candidateOnly": d.get("candidateOnly", True),
        "detail": d,
    }
