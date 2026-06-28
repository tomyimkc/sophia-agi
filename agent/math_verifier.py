# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hard-oracle math verification: sympy canonicalize/compare → accepted/rejected/abstain.

Correctness for curriculum training and RLVR rewards is decided here (and via
``agent.verifiers.math_equivalent``), NOT by the provenance/wisdom gate. Fail-closed
when sympy is unavailable. The Lean 4 backend (``agent.lean_backend``, via LeanDojo,
Path B of the Two-Paths-To-Novelty roadmap) is wired but **opt-in**: when ``lean-dojo``
is not installed (the CI/production default), ``use_lean=True`` abstains fail-closed
with ``lean_unavailable`` — the exact pre-wiring behavior is preserved.
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
    lean_proof: str | None = None,
) -> dict[str, Any]:
    """Compare ``answer`` to ``gold`` with a hard sympy oracle, or — when
    ``use_lean=True`` — verify a Lean 4 proof of ``gold`` via ``lean_backend``.

    Returns ``{verdict, reasons, detail}`` where verdict is one of
    ``accepted``, ``rejected``, ``abstain``.

    Lean path (Path B): when ``use_lean`` and ``lean_proof`` is supplied, delegate to
    ``agent.lean_backend.verify_proof``. When lean-dojo is absent (the default),
    this abstains with ``lean_unavailable`` — fail-closed, never fabricates a verdict.
    """
    if use_lean:
        # Delegate to the Lean 4 backend (opt-in extra; abstains when not installed).
        # Without lean_proof we cannot ask Lean to verify anything -> abstain.
        from agent import lean_backend

        if not lean_backend.lean_available():
            return {
                "verdict": "abstain",
                "reasons": ["lean_unavailable: lean-dojo not installed (opt-in extra)"],
                "detail": {"backend": "lean", "lean": False},
            }
        if not lean_proof:
            return {
                "verdict": "abstain",
                "reasons": ["lean_unavailable: use_lean=True requires lean_proof (a `theorem ... := by ...` block)"],
                "detail": {"backend": "lean", "lean": True, "lean_proof_supplied": False},
            }
        check = lean_backend.verify_proof(theorem=gold, proof=lean_proof)
        out = check.to_dict()
        # Normalize the backend id to "lean" (this verifier's contract) — LeanCheck.to_dict
        # reports "lean4" (the toolchain), but every other branch here and all
        # callers/tests identify this path as "lean". A consistent id avoids breaking
        # downstream consumers that switch on detail.backend.
        (out.get("detail") or {})["backend"] = "lean"
        return out
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
