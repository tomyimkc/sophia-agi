# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Execution / outcome verification — the ground-truth family the wedge needs.

The deliberation-roofline result (``reasoning/deliberation_roofline.py``) says the
*verifier* sets the quality ceiling, not the compute. Citation/symbolic verifiers
verify "is this claim true *about a text*"; they cannot verify "will this *action*
achieve the goal." For the agent domains that define Level-3 (code, tools, repo
repair, planning), ground truth is *machine-checkable by execution*: tests pass or
they don't, a build succeeds or fails, a tool returns a value or an error, a plan's
stated objective is met or it isn't.

Execution verification is a verifier family that needs no citation — it needs a
RESULT. It is strictly more automatable, more reliable, and more κ-stable than
citation verification, which is why raising its status from "scattered individual
verifiers" to "a registered first-class family" raises the verifier ceiling the
roofline bounds capability by.

This module is deliberately THIN: the executable primitives already live in
``agent.verifiers`` (``code_tests_pass``, ``unit_test``, ``arithmetic_sound``).
What was missing is (a) the two AGENT-domain execution verifiers that have no text
analogue — tool-result-validity and plan-completion — and (b) a named FAMILY
registry + a ``verifier_for_task`` dispatcher so the closed loop and the Level-3
lanes can route a task to its execution verifier automatically instead of every
caller hand-wiring one.

Discipline (Sophia, preserved): every verifier is deterministic, offline-testable
where the ground truth is injectable, and fail-closed (a tool error, a missing
code block, or an unmet objective fails the step — never a silent pass). The
family carries ``candidateOnly``/``level3Evidence`` discipline at the report level;
this module is plumbing, not evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from agent.verifiers import Verifier, _fail, _ok, code_tests_pass, unit_test

# --------------------------------------------------------------------------- #
# Agent-domain execution verifiers (no text analogue) — the new primitives
# --------------------------------------------------------------------------- #


def tool_result_validity(*, require_ok: bool = True) -> Verifier:
    """Pass iff every tool result in the step's tool_results is valid.

    A tool that returned an error is execution ground truth that the action
    failed — fail-closed, never a silent pass. ``require_ok=False`` relaxes to
    "the results exist and were observed" (use only when an error itself is an
    acceptable signal, e.g. a probe that expects a 404)."""

    def _verify(text: str, task: Any, step: dict) -> dict:
        results = step.get("tool_results") or step.get("toolResults") or []
        if not results:
            return _fail(["no tool results observed for a tool step"], {"results": 0})
        bad = [r.get("tool", "?") for r in results if not r.get("ok", False)]
        if require_ok and bad:
            return _fail([f"tool(s) returned errors: {', '.join(bad)}"], {"failedTools": bad, "results": len(results)})
        return _ok({"results": len(results), "allOk": not bad})

    return _verify


def plan_completion(*, objective_marker: str = "Decision") -> Verifier:
    """Pass iff the step's stated objective was met.

    For agent tasks the objective is often "produce a decision / a completed
    artifact"; the cheapest machine-checkable proxy is that the structured
    objective marker is present AND the step was reached (not truncated).
    Pair with a harder objective check (test-pass / file-exists) via
    ``all_of`` when the goal is concretely checkable — this is the floor."""

    def _verify(text: str, task: Any, step: dict) -> dict:
        if not str(text or "").strip():
            return _fail(["empty output — objective not met"])
        if objective_marker and objective_marker.lower() not in (text or "").lower():
            return _fail([f"objective marker '{objective_marker}' absent — plan did not complete"],
                         {"objectiveMarker": objective_marker})
        return _ok({"objectiveMarker": objective_marker})

    return _verify


# --------------------------------------------------------------------------- #
# The execution/outcome FAMILY registry — named, first-class, dispatchable
# --------------------------------------------------------------------------- #
# Ground-truth family: pass conditions are determined by RUNNING something, not by
# reading text. Re-exported from agent.verifiers where they already lived, plus the
# two new agent-domain primitives above. Each entry is a parameterless factory so
# it slots into the existing CLI/MCP registry contract.

EXECUTION_VERIFIERS: dict[str, Callable[[], Verifier]] = {
    "code_tests_pass": code_tests_pass,        # run extracted code, exit 0
    "unit_test": lambda: unit_test(["python", "-m", "pytest", "-q"]),  # run a suite
    "tool_result_validity": tool_result_validity,
    "plan_completion": plan_completion,
}


@dataclass(frozen=True)
class TaskSpec:
    """A routing hint for ``verifier_for_task``. ``kind`` picks the execution
    family member; ``kind=None`` means "no executable ground truth — fall back to
    the caller's default (e.g. the epistemic gate / citation verifier)." The
    roofline predicts you should weight executable-truth domains where you can."""

    kind: str | None  # "code" | "tool" | "plan" | None
    objective: str = ""


def verifier_for_task(spec: TaskSpec, *, objective_marker: str = "Decision") -> Verifier | None:
    """Dispatch a task to its execution verifier, or None if none applies.

    Returns ``None`` (not the gate) so the caller can compose: a coding case
    returns ``code_tests_pass``; an agent tool-use case returns
    ``all_of(tool_result_validity(), plan_completion())``; a pure-prose case
    returns ``None`` and the caller keeps its citation/gate default. This is the
    wedge that makes execution verification FIRST-CLASS: one call routes by task
    kind instead of every caller hardcoding its verifier."""
    if spec.kind == "code":
        return code_tests_pass()
    if spec.kind == "tool":
        from agent.verifiers import all_of

        return all_of(tool_result_validity(), plan_completion(objective_marker=objective_marker))
    if spec.kind == "plan":
        return plan_completion(objective_marker=objective_marker)
    return None


def family_report() -> dict[str, Any]:
    """Introspection artifact: which execution verifiers are registered, with the
    one-line semantics of each. Candidate-only — documents the family, not evidence."""
    semantics = {
        "code_tests_pass": "run extracted code in an isolated tempdir; pass iff exit 0",
        "unit_test": "run a command (pytest/build) and pass iff exit 0; world state is the verifier",
        "tool_result_validity": "pass iff every tool result in the step is ok (no tool errors)",
        "plan_completion": "pass iff the objective marker is present (floor; pair with a harder check)",
    }
    return {
        "schema": "sophia.execution_verifiers.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "family": "execution_outcome",
        "thesis": (
            "The verifier sets the ceiling (deliberation_roofline). Execution truth "
            "(test-pass / build-success / tool-result / plan-completion) is the highest-"
            "fidelity, most κ-stable verifier family for agent domains — raising it to "
            "first-class raises the ceiling the loop and the Level-3 lanes run under."
        ),
        "members": [{"name": k, "semantics": semantics.get(k, "")} for k in EXECUTION_VERIFIERS],
        "count": len(EXECUTION_VERIFIERS),
    }


__all__ = [
    "EXECUTION_VERIFIERS",
    "TaskSpec",
    "tool_result_validity",
    "plan_completion",
    "verifier_for_task",
    "family_report",
]
