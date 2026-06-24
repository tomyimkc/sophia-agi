"""Skill: provenance_fact_check — check a statement for attribution/provenance violations."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "provenance_fact_check",
    summary="Check a statement for attribution/provenance violations; abstains rather than fabricate.",
    uses=("check_claim",),
)
def provenance_fact_check(*, text: str) -> dict:
    res = call("check_claim", text=text)
    passed = bool(res.get("passed"))
    return {
        "verdict": "ok" if passed else "flagged",
        "passed": passed,
        "violations": res.get("violations", []),
        "reasons": res.get("reasons", []),
        "detail": res,
    }
