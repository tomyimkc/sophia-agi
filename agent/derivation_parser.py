# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Parse a model's free-text working into an ordered list of :class:`Step`.

The bridge between a language model and :mod:`agent.step_verifier`: a model emits
a derivation as prose, and this turns it into the structured chain the verifier
checks. Three formats are accepted, in priority order:

1. **Explicit step lines** — ``STEP: <expr> | <rule>`` (the format the solver
   prompt asks for). Most reliable.
2. **Equality chains** — ``a = b = c`` on a single line is split into the
   consecutive expressions ``a``, ``b``, ``c`` (each claimed equal to the prior).
3. **Answer-only fallback** — if neither is present, the final answer is
   extracted (``\\boxed{}`` / "answer is …" / last line) as a single step, so the
   final-answer oracle can still check it (Verified-Step Coverage will be low,
   which honestly reflects an unshown derivation).

Pure-Python, deterministic, no model call.
"""

from __future__ import annotations

import re

from agent.step_verifier import Domain, Step
from agent.verifiers import extract_math_answer

_STEP_LINE = re.compile(r"^\s*STEP\s*[:\-]\s*(.+)$", re.IGNORECASE)
# Split on a single '=' that is not part of ==, <=, >=, != .
_SINGLE_EQ = re.compile(r"(?<![<>=!])=(?![=])")


def _looks_like_expr(s: str) -> bool:
    """A filter for a derivation atom: it must carry a digit or a math operator
    (so a quantity like ``30 N`` or an expression like ``x+1`` qualifies, but a
    bare prose fragment like ``I have no idea`` does not)."""
    s = s.strip()
    return bool(s) and bool(re.search(r"\d|\*\*|[+\-*/^]", s))


def parse_derivation(text: str, *, domain: Domain = "math") -> list[Step]:
    """Parse ``text`` into a list of :class:`Step` for the given ``domain``."""
    if not text:
        return []

    # 1. Explicit STEP: lines.
    steps: list[Step] = []
    for line in text.splitlines():
        m = _STEP_LINE.match(line)
        if not m:
            continue
        expr, _, rule = m.group(1).partition("|")
        if expr.strip():
            steps.append(Step(expr.strip(), rule=rule.strip(), domain=domain))
    if steps:
        return steps

    # 2. Equality chain on a single line (prefer the longest chain).
    best: list[str] = []
    for line in text.splitlines():
        parts = [p.strip() for p in _SINGLE_EQ.split(line)]
        parts = [p for p in parts if _looks_like_expr(p)]
        if len(parts) > len(best):
            best = parts
    if len(best) >= 2:
        return [Step(p, domain=domain) for p in best]

    # 3. Answer-only fallback.
    ans = extract_math_answer(text)
    if ans and _looks_like_expr(ans):
        return [Step(ans.strip(), rule="final answer (no derivation shown)", domain=domain)]
    return []
