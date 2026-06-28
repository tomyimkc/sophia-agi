# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Inference-time council-panel format templates (no weights, no retrain)."""

from __future__ import annotations

from agent.council_deliberate import Deliberation

TEAM_AGENTS_SYSTEM = (
    "You are a source-disciplined team-agents advisor. Decompose the question across "
    "relevant expert seats, state each seat's finding with a source where one is relied on, "
    "then give one synthesised decision. If seats legitimately diverge on values or risk "
    "appetite, ABSTAIN and flag the conflict rather than invent consensus. If you cannot "
    "verify a needed authority or figure, ABSTAIN and route to a human. Label clearly as "
    "not professional advice; end with a 中文摘要."
)

RELIGION_COUNCIL_SYSTEM = (
    "You are a source-disciplined council advisor. For religion questions, begin with "
    "'**Council panel:**' and separate theological, historical-critical, comparative, "
    "and tradition-specific voices. Keep traditions distinct, state uncertainty, and end "
    "with a 中文摘要."
)

def render_team_target(d: Deliberation) -> tuple[str, str]:
    """Return (kind, assistant_text) for team-agents distillation / eval targets."""
    clean = [s for s in d.seats if s.ok and s.gatePassed]
    if not clean and "insufficient" in (d.synthesis or "").lower():
        return "abstention", d.synthesis.strip()
    perspectives = "\n".join(f"- {s.displayName}: {s.answer}" for s in clean)
    body = (
        f"Perspectives:\n{perspectives}\n\nDecision: {d.synthesis.strip()}"
        if perspectives
        else d.synthesis.strip()
    )
    kind = "abstention" if "insufficient" in body.lower() or "conflict" in body.lower() else "trace"
    return kind, body


__all__ = ["RELIGION_COUNCIL_SYSTEM", "TEAM_AGENTS_SYSTEM", "render_team_target"]
