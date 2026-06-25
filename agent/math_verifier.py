# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hard-oracle math verification: sympy canonicalize/compare → accepted/rejected/abstain.

Correctness for curriculum training and RLVR rewards is decided here (and via
``agent.verifiers.math_equivalent``), NOT by the provenance/wisdom gate. Fail-closed
when sympy is unavailable. Optional Lean backend is reserved but not wired — requests
return ``abstain`` with ``lean_unavailable``.
"""
from __future__ import annotations

from typing import Any, Literal

from agent.verifiers import extract_math_answer, math_equivalent

Verdict = Literal["accepted", "rejected", "abstain"]


def sympy_available() -> bool:
    try:
        import sympy  # noqa: F401
        return True
    except Exception:
        return False


def canonicalize(expr: str) -> tuple[bool, str | None]:
    """Return (ok, canonical_str). ``ok=False`` when sympy cannot parse."""
    if not sympy_available():
        return False, None
    from agent.verifiers import _sympy_parse  # noqa: PLC2701

    ok, parsed = _sympy_parse(expr)
    if not ok or parsed is None:
        return False, None
    import sympy as sp

    return True, str(sp.simplify(parsed))


def verify(
    answer: str,
    gold: str,
    *,
    extract: bool = True,
    use_lean: bool = False,
) -> dict[str, Any]:
    """Compare ``answer`` to ``gold`` with a hard sympy oracle.

    Returns ``{verdict, reasons, detail}`` where verdict is one of
    ``accepted``, ``rejected``, ``abstain``.
    """
    if use_lean:
        return {
            "verdict": "abstain",
            "reasons": ["lean_unavailable: optional Lean backend not configured"],
            "detail": {"backend": "lean", "lean": False},
        }
    if not sympy_available():
        return {
            "verdict": "abstain",
            "reasons": ["sympy_unavailable: cannot verify math equivalence"],
            "detail": {"sympy": False, "gold": gold},
        }
    res = math_equivalent(gold, extract=extract)(answer or "", None, {})
    if res["passed"]:
        got = (res.get("detail") or {}).get("got") or extract_math_answer(answer) if extract else answer
        return {
            "verdict": "accepted",
            "reasons": [],
            "detail": {"sympy": True, "gold": gold, "got": got},
        }
    reasons = list(res.get("reasons") or ["not equivalent"])
    if any("sympy_unavailable" in r for r in reasons):
        return {"verdict": "abstain", "reasons": reasons, "detail": {"sympy": False}}
    if any("unparseable" in r or "no math answer" in r for r in reasons):
        return {
            "verdict": "abstain",
            "reasons": reasons,
            "detail": {"sympy": True, "gold": gold},
        }
    return {
        "verdict": "rejected",
        "reasons": reasons,
        "detail": {"sympy": True, "gold": gold},
    }
