"""Skill: moral_parliament_decide — bounded moral-uncertainty aggregation."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "moral_parliament_decide",
    summary="Aggregate a moral decision across multiple ethical theories (bounded moral uncertainty); escalates on high disagreement.",
    uses=("moral_parliament_tool",),
)
def moral_parliament_decide(*, text: str, context: dict | None = None) -> dict:
    res = call("moral_parliament_tool", text=text, context=context)
    return {
        "verdict": res.get("verdict", "held"),
        "aggregate": res.get("aggregate"),
        "variance": res.get("variance"),
        "reason": res.get("reason"),
        "detail": res,
    }
