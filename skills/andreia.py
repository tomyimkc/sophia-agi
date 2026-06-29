# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: courage_advocate — Andreia courage gate, fail-closed.

The dual of ``conscience_abstain``: where the conscience skills decide whether to
*hold back*, this decides whether the brave, well-calibrated move is to act. It
never overrides a hard prohibition, and on any error it abstains (``held``).
"""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "courage_advocate",
    summary="Assess whether to act|escalate|hold despite fear (Andreia courage gate). Flags cowardice disguised as prudence; never overrides a hard prohibition.",
    uses=("courage_assess_tool", "cowardice_check_tool"),
)
def courage_advocate(*, text: str, context: dict | None = None, samples: list | None = None) -> dict:
    d = call("courage_assess_tool", text=text, samples=samples, context=context)
    return {
        "verdict": d.get("verdict", "hold"),
        "cq": d.get("cq"),
        "reason": d.get("reason"),
        "forces": d.get("forces", {}),
        "fearAttribution": d.get("fearAttribution", {}),
        "cowardice": d.get("cowardice", {}),
        "blockRespected": d.get("blockRespected", False),
        "candidateOnly": d.get("candidateOnly", True),
        "detail": d,
    }
