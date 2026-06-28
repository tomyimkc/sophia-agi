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
    use_lean: bool = False,
    lean_proof: str | None = None,
) -> dict[str, Any]:
    """Compare ``answer`` to ``gold`` with the dimensional/numeric oracle, or — when
    ``use_lean=True`` — verify a Lean 4 proof of ``gold`` via ``lean_backend``.

    Returns ``{verdict, reasons, detail}`` where verdict is one of ``accepted``,
    ``rejected``, ``abstain``. ``abstain`` arises for a *symbolic* gold when sympy is
    absent, or on the Lean path when lean-dojo is not installed — fail-closed, never
    a guess. A dimension mismatch or an out-of-tolerance value is a hard ``rejected``.

    Lean path (the formal-physics on-ramp: PhysLib / Lean4Physics / LeanPhysBench):
    when ``use_lean`` and ``lean_proof`` is supplied, delegate to
    ``agent.lean_backend.verify_proof``. When lean-dojo is absent (the CI/default),
    this abstains with ``lean_unavailable`` — the exact opt-in/abstain contract the
    math verifier uses, so a physics theorem can be *formally* checked once a Lean
    physics library is wired, with no behavior change until then.
    """
    if use_lean:
        from agent import lean_backend

        if not lean_backend.lean_available():
            return {"verdict": "abstain",
                    "reasons": ["lean_unavailable: lean-dojo not installed (opt-in extra)"],
                    "detail": {"backend": "lean", "lean": False}}
        if not lean_proof:
            return {"verdict": "abstain",
                    "reasons": ["lean_unavailable: use_lean=True requires lean_proof"],
                    "detail": {"backend": "lean", "lean": True, "lean_proof_supplied": False}}
        out = lean_backend.verify_proof(theorem=gold, proof=lean_proof).to_dict()
        # to_dict() already yields {verdict, reasons, detail}; pass it through.
        return {"verdict": out["verdict"], "reasons": out["reasons"], "detail": out["detail"]}

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
