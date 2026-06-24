# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: council_adjudicate — gate-filtered multi-seat council deliberation."""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


def _seat_id(seat) -> object:
    return seat.get("seatId", seat.get("id")) if isinstance(seat, dict) else seat


@sophia_skill(
    "council_adjudicate",
    summary="Run a multi-seat council deliberation with per-seat gating; returns the gated synthesis (informational, not advice).",
    uses=("council_deliberate",),
)
def council_adjudicate(*, query: str, model: str = "mock") -> dict:
    res = call("council_deliberate", query=query, model=model)
    return {
        "verdict": "deliberated",
        "synthesis": res.get("synthesis"),
        "seats": [_seat_id(s) for s in (res.get("seats") or [])],
        "gatedOutSeatIds": res.get("gatedOutSeatIds", []),
        "notAdvice": res.get("notAdvice", True),
        "detail": res,
    }
