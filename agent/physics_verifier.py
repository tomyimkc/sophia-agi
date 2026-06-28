# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hard-oracle physics verification: dimensional analysis + numeric tolerance →
accepted/rejected/abstain.

The physics analogue of ``agent.math_verifier``. Correctness for curriculum
training and RLVR rewards is decided here (and via ``agent.verifiers.physics_equivalent``),
NOT by the provenance/wisdom gate. The ground truth is dimensional: a numeric
answer must match the gold's SI dimension *and* value — ``9.8 J`` is not
``9.8 m/s^2``. Units are pure-Python (``agent.units``), so this never needs a GPU
or an optional backend and always runs in CI.

Symbolic golds (closed-form, no number) fall back to the sympy oracle via
``physics_equivalent`` → ``math_equivalent``; when sympy is unavailable that path
abstains fail-closed (``sympy_unavailable``), never fabricating a verdict.
"""

from __future__ import annotations

from typing import Any, Literal

from agent.units import parse_quantity
from agent.verifiers import physics_equivalent

Verdict = Literal["accepted", "rejected", "abstain"]


def units_available() -> bool:
    """The units engine is stdlib-only — always available (kept for symmetry)."""
    return True


def verify(
    answer: str,
    gold: str,
    *,
    rtol: float = 1e-2,
    extract: bool = True,
) -> dict[str, Any]:
    """Compare ``answer`` to ``gold`` with the dimensional/numeric oracle.

    Returns ``{verdict, reasons, detail}`` where verdict is one of ``accepted``,
    ``rejected``, ``abstain``. ``abstain`` arises only for a *symbolic* gold when
    sympy is absent (the algebra cannot be checked) — fail-closed, never a guess.
    A dimension mismatch or an out-of-tolerance value is a hard ``rejected``.
    """
    res = physics_equivalent(gold, rtol=rtol, extract=extract)(answer or "", None, {})
    detail = dict(res.get("detail") or {})
    reasons = list(res.get("reasons") or [])
    if res.get("passed"):
        return {"verdict": "accepted", "reasons": [], "detail": detail}
    # A symbolic gold with no sympy yields the math_equivalent held verdict.
    ok_g, _, _ = parse_quantity(gold)
    if not ok_g and any("sympy_unavailable" in r for r in reasons):
        return {"verdict": "abstain", "reasons": reasons, "detail": detail}
    return {"verdict": "rejected", "reasons": reasons, "detail": detail}
