# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: deception_scan — black-box deception signals + stated confidence."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "deception_scan",
    summary="Flag deception signals: confidence/evidence mismatch, claiming verification while the gate is held, source laundering, gate tampering.",
    uses=("deception_check_tool", "uncertainty_score"),
)
def deception_scan(*, text: str, context: dict | None = None) -> dict:
    d = call("deception_check_tool", text=text, context=context)
    u = call("uncertainty_score", text=text)
    return {
        "verdict": d.get("verdict", "held"),
        "risk": d.get("risk"),
        "signals": d.get("signals", []),
        "reason": d.get("reason"),
        "statedConfidence": u.get("confidence"),
        "detail": {"deception": d, "uncertainty": u},
    }
