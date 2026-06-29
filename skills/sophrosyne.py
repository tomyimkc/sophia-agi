# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: measure_advocate — Sophrosyne temperance gate, fail-closed.

The magnitude-axis sibling of ``courage_advocate``: where Andreia decides the
*direction* (act vs hold) and the conscience skills decide *truth*, this decides
*how much* — whether to hold the mean, restrain (excess), sustain (deficiency), or
escalate. It never suppresses a required verification step, and on any error it
returns the conservative ``proportionate``.
"""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "measure_advocate",
    summary="Assess whether to hold the mean|restrain|sustain|escalate (Sophrosyne temperance gate). Flags excess (verbosity/over-hedging/runaway loops) and deficiency (premature stop/truncation); never cuts a required verification step.",
    uses=("temperance_assess_tool", "intemperance_check_tool"),
)
def measure_advocate(*, text: str, context: dict | None = None) -> dict:
    d = call("temperance_assess_tool", text=text, context=context)
    return {
        "verdict": d.get("verdict", "proportionate"),
        "mq": d.get("mq"),
        "reason": d.get("reason"),
        "forces": d.get("forces", {}),
        "intemperance": d.get("intemperance", {}),
        "stepRespected": d.get("stepRespected", False),
        "candidateOnly": d.get("candidateOnly", True),
        "detail": d,
    }
