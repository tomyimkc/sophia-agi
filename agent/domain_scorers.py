# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Domain-pluggable scoring registry (Hurdle 2 — broad transfer).

Lets the IDENTICAL eval ladder + promotion loop score structurally different task
families through their own SOUND verifiers, instead of assuming provenance. The
honest transfer test is: does the same retention/promotion machinery help on
families whose ground truth is *not* a citation corpus?

A scorer is ``score_fn(case, response, ctx) -> (ok: bool, reasons: list[str])`` —
the exact shape ``agent.benchmark_checks.score_case`` already returns, so the
eval backends dispatch through here with no change to their reporting.

Registered kinds (all deterministic, offline, no model):
- ``provenance`` — wraps ``benchmark_checks.score_case`` (corpus-grounded attribution traps).
- ``math``       — extracts the stated final answer and checks numeric equality, with
                   ``verifiers.arithmetic_sound`` as a soundness veto (no false stated equality).
- ``coding``     — wraps ``verifiers.code_tests_pass`` (extracts + executes Python, exit code).

A case is routed by its ``"kind"`` field when present, else by the domain's default
kind (`DOMAIN_KIND`). Unknown domains fall back to ``provenance`` for back-compat.
"""

from __future__ import annotations

import re
from typing import Callable

# (ok, reasons)
ScoreFn = Callable[[dict, str, dict], "tuple[bool, list[str]]"]

# Default family for each domain. Provenance domains are corpus-grounded; math and
# coding are the structurally-different, sound-verifier families used to test transfer.
DOMAIN_KIND: dict[str, str] = {
    "philosophy": "provenance",
    "psychology": "provenance",
    "history": "provenance",
    "religion": "provenance",
    "personality": "provenance",
    "math": "math",
    "coding": "coding",
}

# ``answer = 42`` / ``answer: 42`` (the format the math pack prompts for).
_ANSWER_RE = re.compile(r"answer\s*[=:]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
# Fallback: any signed decimal token, so we can read the last number stated.
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _provenance_scorer(case: dict, response: str, ctx: dict) -> "tuple[bool, list[str]]":
    from agent.benchmark_checks import score_case

    return score_case(case, response, ctx.get("traditions", {}) or {})


def _extract_number(text: str) -> float | None:
    m = _ANSWER_RE.search(text)
    if m:
        return float(m.group(1))
    nums = _NUM_RE.findall(text)
    return float(nums[-1]) if nums else None


def _math_scorer(case: dict, response: str, ctx: dict) -> "tuple[bool, list[str]]":
    """Sound for well-defined numeric problems: exact answer-match + no false equality.

    Pass requires BOTH (a) the stated final answer equals the expected value within
    tolerance, and (b) the response contains no FALSE arithmetic equality anywhere
    (e.g. ``2 + 2 = 5`` in the working). (b) is a soundness veto, not a presence
    requirement — text with no checkable arithmetic does not fail on (b) alone.
    """
    from agent.verifiers import arithmetic_sound

    expected = case.get("expectedAnswer", case.get("answer"))
    tol = float(case.get("tol", 1e-6))
    reasons: list[str] = []
    ok = True

    if expected is None:
        return False, ["math case missing expectedAnswer/answer"]

    got = _extract_number(response)
    if got is None:
        ok = False
        reasons.append("no numeric answer found in response")
    elif abs(got - float(expected)) > tol:
        ok = False
        reasons.append(f"answer {got:g} != expected {float(expected):g}")

    sound = arithmetic_sound(tol=tol)(response, case, {})
    if not sound.get("passed", True):
        ok = False
        reasons.extend(sound.get("reasons", ["false arithmetic in response"]))

    return ok, reasons


def _coding_scorer(case: dict, response: str, ctx: dict) -> "tuple[bool, list[str]]":
    """Wrap the executable code verifier. The environment decides correctness.

    ``ctx['allow_execution']`` (default True) lets callers force syntax-only for
    sandboxes that forbid execution; the per-case ``timeoutSec`` bounds runtime.
    """
    from agent.verifiers import code_tests_pass

    allow_execution = bool(ctx.get("allow_execution", True))
    timeout = int(case.get("timeoutSec", 30))
    result = code_tests_pass(timeout_sec=timeout, allow_execution=allow_execution)(response, case, {})
    return bool(result.get("passed", False)), list(result.get("reasons", []))


KIND_SCORERS: dict[str, ScoreFn] = {
    "provenance": _provenance_scorer,
    "math": _math_scorer,
    "coding": _coding_scorer,
}


def kind_for(domain: str, case: dict | None = None) -> str:
    """Resolve the scorer kind: explicit case ``kind`` wins, else the domain default."""
    if case is not None:
        explicit = case.get("kind")
        if explicit:
            return str(explicit)
    return DOMAIN_KIND.get(domain, "provenance")


def score_for_domain(
    domain: str,
    case: dict,
    response: str,
    *,
    ctx: dict | None = None,
) -> "tuple[bool, list[str]]":
    """Dispatch a single case to the right family scorer.

    This is the one call both eval backends use, so the identical ladder scores
    provenance, math, and coding through their own sound verifiers.
    """
    kind = kind_for(domain, case)
    scorer = KIND_SCORERS.get(kind)
    if scorer is None:
        return False, [f"no scorer registered for kind '{kind}' (domain '{domain}')"]
    return scorer(case, response, ctx or {})


def register_kind(kind: str, scorer: ScoreFn) -> None:
    """Register a new task-family scorer (e.g. logic, planning) at import time."""
    KIND_SCORERS[kind] = scorer
