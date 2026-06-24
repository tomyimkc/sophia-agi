"""Skill: moral_public_standard_review — Moral Gate v2 (public standard + parliament)."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "moral_public_standard_review",
    summary="Moral Gate v2: screen text against the public moral standard (overlapping consensus) and the moral parliament.",
    uses=("public_standard_check_tool", "moral_parliament_tool"),
)
def moral_public_standard_review(*, text: str, context: dict | None = None) -> dict:
    ps = call("public_standard_check_tool", text=text, context=context)
    mp = call("moral_parliament_tool", text=text, context=context)
    return {
        "verdict": ps.get("verdict", "held"),
        "isNormative": ps.get("isNormative"),
        "violations": ps.get("violations", []),
        "unmetDuties": ps.get("unmetDuties", []),
        "grayZone": ps.get("grayZone"),
        "parliamentVerdict": mp.get("verdict"),
        "parliamentAggregate": mp.get("aggregate"),
        "detail": {"publicStandard": ps, "parliament": mp},
    }
