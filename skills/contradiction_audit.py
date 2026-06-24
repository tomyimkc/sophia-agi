"""Skill: contradiction_audit — surface internal contradictions in the OKF knowledge base."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "contradiction_audit",
    summary="Audit the OKF knowledge base for declared contradictions, self/tradition merges, supersede cycles, and confidence laundering.",
    uses=("wiki_contradictions",),
)
def contradiction_audit() -> dict:
    c = call("wiki_contradictions")
    counts = {k: (len(v) if isinstance(v, list) else v) for k, v in c.items()}
    total = sum(n for n in counts.values() if isinstance(n, int))
    return {
        "verdict": "clean" if total == 0 else "contradictions-found",
        "totalIssues": total,
        "counts": counts,
        "detail": c,
    }
