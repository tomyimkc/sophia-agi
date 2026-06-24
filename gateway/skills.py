# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifiable Skills (P2): a skill = program + verifier + classification + eval suite.

A skill is registered like any tool, but it carries its own ``verifier_ref`` (its output
is checked before it ships) and an ``eval_suite`` so it can SELF-TEST — the basis for the
reliability registry (P3) and the per-skill flywheel (P4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from gateway.registry import ToolEntry


@dataclass
class SkillEntry:
    skill_id: str
    program: "Callable[[dict], object]"          # args -> output (the skill's work)
    verifier_ref: str = "none"                    # how the output is checked
    blp_level: str = "UNCLASSIFIED"
    side_effects: str = "none"
    allowed_roles: "frozenset | None" = None
    description: str = ""
    eval_suite: "list[dict]" = field(default_factory=list)  # [{args, expect}] for self-test

    def to_tool(self) -> "ToolEntry":
        return ToolEntry(
            id=self.skill_id, handler=self.program, kind="skill",
            verifier_ref=self.verifier_ref, blp_level=self.blp_level,
            side_effects=self.side_effects, allowed_roles=self.allowed_roles,
            description=self.description or f"verifiable skill {self.skill_id}",
        )


def eval_skill(gateway, skill: "SkillEntry", *, role: "str | None" = None,
               clearance: str = "UNCLASSIFIED") -> dict:
    """Run the skill's eval suite through the gateway (so each output is gated) and report
    the accept-rate vs the expected results — the skill's self-test."""
    if not skill.eval_suite:
        return {"skill_id": skill.skill_id, "evaluated": 0, "acceptRate": None, "correct": None}
    accepted = correct = 0
    for case in skill.eval_suite:
        resp = gateway.call_tool(skill.skill_id, case.get("args", {}), role=role, clearance=clearance)
        ok = resp.get("verdict") == "accepted"
        accepted += int(ok)
        if "expect" in case and ok:
            correct += int(resp.get("result") == case["expect"] or
                           (isinstance(resp.get("result"), dict) and
                            resp["result"].get("answer") == case["expect"]))
    n = len(skill.eval_suite)
    return {
        "skill_id": skill.skill_id, "evaluated": n,
        "acceptRate": round(accepted / n, 4),
        "correct": round(correct / n, 4) if any("expect" in c for c in skill.eval_suite) else None,
    }
