# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: fairness_advocate — Dikaiosyne justice gate, fail-closed.

The relational-axis sibling of ``courage_advocate`` and ``measure_advocate``: where
those judge a single decision, this judges consistency ACROSS cases — whether like
cases are treated alike (impartial) or the verdict flips on a morally irrelevant
feature (partial), or a relevant difference is ignored (false_equivalence). It never
endorses false balance, and on any error it returns the conservative ``impartial``.
"""
from __future__ import annotations

from skills.core import sophia_skill
from skills.mcp_bridge import call


@sophia_skill(
    "fairness_advocate",
    summary="Audit impartiality: treat like cases alike (Dikaiosyne justice gate, Role A). Flags partiality (verdict flips on an irrelevant feature) and false balance; never endorses equal time for a prohibited claim.",
    uses=("justice_assess_tool", "partiality_check_tool"),
)
def fairness_advocate(*, text: str = "", irrelevant_class: list | None = None,
                      relevant_class: list | None = None, context: dict | None = None) -> dict:
    d = call("justice_assess_tool", text=text, irrelevant_class=irrelevant_class,
             relevant_class=relevant_class, context=context)
    return {
        "verdict": d.get("verdict", "impartial"),
        "jq": d.get("jq"),
        "reason": d.get("reason"),
        "detail": d.get("detail", {}),
        "partiality": d.get("partiality", {}),
        "blockRespected": d.get("blockRespected", False),
        "candidateOnly": d.get("candidateOnly", True),
        "full": d,
    }
