# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: conscience_abstain — run output through the Conscience Kernel."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "conscience_abstain",
    summary="Run output/tool/memory text through the Conscience Kernel; returns allow/revise/retrieve/clarify/escalate/abstain/block.",
    uses=("conscience_check_tool", "uncertainty_score"),
)
def conscience_abstain(*, text: str, mode: str = "output") -> dict:
    res = call("conscience_check_tool", text=text, mode=mode)
    unc = call("uncertainty_score", text=text)
    return {
        "verdict": res.get("verdict", "held"),
        "action": res.get("action"),
        "reason": res.get("reason"),
        "recommendedActions": res.get("recommendedActions", []),
        "uncertaintyType": unc.get("uncertaintyType"),
        "confidence": unc.get("confidence"),
        "detail": res,
    }
