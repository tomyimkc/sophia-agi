# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: source_discipline_enforce — fail-closed source-discipline gate."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "source_discipline_enforce",
    summary="Fail-closed source-discipline gate: allow only when there are no attribution/provenance violations.",
    uses=("check_claim",),
)
def source_discipline_enforce(*, text: str) -> dict:
    res = call("check_claim", text=text)
    passed = bool(res.get("passed"))
    return {
        "verdict": "allow" if passed else "block",
        "allowed": passed,
        "violations": res.get("violations", []),
        "detail": res,
    }
