# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: agent_trajectory_eval — score an agent run for mid-plan faithfulness."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "agent_trajectory_eval",
    summary="Score a whole agent trajectory step by step; abstains rather than certify a run "
    "with an ungrounded or unverifiable step.",
    uses=("trajectory_eval",),
)
def agent_trajectory_eval(*, trajectory: list) -> dict:
    res = call("trajectory_eval", trajectory=trajectory)
    if res.get("error"):
        return {"verdict": "held", "ok": False, "reason": res["error"]}
    verdict = res.get("verdict")
    # Map the evaluator's fail-closed verdicts onto the skill layer's contract:
    # only an "accept" run is ok; everything else is held (abstain) or flagged.
    if verdict == "accept":
        skill_verdict, ok = "ok", True
    elif verdict == "blocked":
        skill_verdict, ok = "flagged", False
    else:
        skill_verdict, ok = "held", False
    return {
        "verdict": skill_verdict,
        "ok": ok,
        "trajectoryVerdict": verdict,
        "faithfulnessScore": res.get("faithfulnessScore"),
        "firstUnfaithfulStep": res.get("firstUnfaithfulStep"),
        "reasons": res.get("reasons", []),
        "detail": res,
    }
