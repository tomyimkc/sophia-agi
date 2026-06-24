# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: belief_revision_explore — OKF belief-graph lookup + counterfactual."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "belief_revision_explore",
    summary="Query the OKF belief graph for an entity and optionally explore a counterfactual (what changes if a source is removed). Abstains when the entity is unknown.",
    uses=("belief", "counterfactual"),
)
def belief_revision_explore(*, entity: str, counterfactual_source: str | None = None) -> dict:
    b = call("belief", entity=entity)
    if not b.get("found"):
        return {
            "verdict": "held",
            "found": False,
            "reason": f"no belief-graph entry for {entity!r}; abstaining",
            "detail": b,
        }
    out = {
        "verdict": "ok",
        "found": True,
        "attributedAuthor": b.get("attributedAuthor"),
        "doNotAttributeTo": b.get("doNotAttributeTo"),
        "effectiveConfidenceRank": b.get("effectiveConfidenceRank"),
        "confidenceLaundered": b.get("confidenceLaundered"),
        "contradicts": b.get("contradicts", []),
        "supersededBy": b.get("supersededBy"),
        "detail": b,
    }
    if counterfactual_source:
        out["counterfactual"] = call("counterfactual", source=counterfactual_source, query=entity)
    return out
