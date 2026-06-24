# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Neurosymbolic formal verifier with an optional Z3 backend (Stage C).

Sophia's gates are strong on *provenance* and *grounding* but had no path for
**formal, solver-checked** consistency — the highest-assurance tier for claims
whose form admits formalization (lattice ordering, contradictory assertions,
temporal ordering). This module adds that tier.

Design constraints (repo discipline):
  - **Optional dependency, fail-closed.** ``z3-solver`` is NOT a hard dependency.
    When z3 is absent, the verifier returns ``status="z3_unavailable"`` with
    verdict ``held`` — it NEVER silently returns ``accepted``. A pure-Python
    fallback handles the small, decidable checks deterministically so CI passes
    offline without z3.
  - **Deterministic + offline.** No network, no model calls.
  - **Honest scope.** This proves *internal logical consistency* of a claim set
    against declared constraints; it does not adjudicate empirical truth.

Result shape (a dict, like the rest of the harness verifiers):
    {
      "verdict": "accepted" | "rejected" | "held",
      "backend": "z3" | "fallback" | "none",
      "status": "consistent" | "contradiction" | "z3_unavailable" | "error",
      "reasons": [...],
      "model": {...} | None,          # a satisfying assignment when consistent
    }
"""

from __future__ import annotations

from typing import Any


def z3_available() -> bool:
    """True iff the optional z3 backend can be imported."""
    try:
        import z3  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------------- #
def check_lattice_consistency(assignments: "dict[str, int]",
                              constraints: "list[tuple[str, str, str]]") -> dict:
    """Check a set of ordering constraints over integer-ranked labels.

    ``assignments`` maps a variable name to its rank (e.g. a BLP/Biba level rank).
    ``constraints`` is a list of ``(lhs, op, rhs)`` where op in {"<=", "<", ">=",
    ">", "==", "!="} and lhs/rhs are variable names. Used to verify, e.g., that a
    derived claim's confidentiality dominates its parents' (no-write-down).

    Prefers z3 when available; otherwise evaluates deterministically in Python
    (these constraints are linear/decidable, so the fallback is exact, not weaker).
    """
    if z3_available():
        return _z3_lattice(assignments, constraints)
    return _fallback_lattice(assignments, constraints)


def check_no_contradiction(claims: "list[dict]") -> dict:
    """Detect direct contradictions in a set of atomic claims.

    Each claim is ``{"subject": s, "predicate": p, "object": o, "negated": bool}``.
    A contradiction is the same (s, p, o) asserted both positively and negatively.
    This is the formal companion to the provenance ``doNotAttributeTo`` rule:
    it catches "X authored Y" and "X did NOT author Y" co-asserted.

    Uses z3 boolean reasoning when available; otherwise an exact set check.
    """
    if z3_available():
        return _z3_contradiction(claims)
    return _fallback_contradiction(claims)


def require_z3(check_fn, *args, **kwargs) -> dict:
    """Run a check that MUST use z3. If z3 is unavailable, fail closed with
    ``held`` / ``z3_unavailable`` rather than falling back. Use this for claims
    where only a real solver result may be trusted."""
    if not z3_available():
        return {
            "verdict": "held",
            "backend": "none",
            "status": "z3_unavailable",
            "reasons": ["z3-solver not installed; install `z3-solver` to enable "
                        "solver-checked formal verification (held, not accepted)"],
            "model": None,
        }
    return check_fn(*args, **kwargs)


# ----------------------------------------------------------------------------- #
# z3 backend
# ----------------------------------------------------------------------------- #
def _z3_lattice(assignments, constraints) -> dict:
    import z3

    s = z3.Solver()
    vs = {name: z3.Int(name) for name in assignments}
    for name, rank in assignments.items():
        s.add(vs[name] == int(rank))
    ops = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
           ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    for lhs, op, rhs in constraints:
        if op not in ops:
            return _err(f"unknown operator {op!r}")
        a = vs.get(lhs, z3.IntVal(_as_int(lhs)))
        b = vs.get(rhs, z3.IntVal(_as_int(rhs)))
        s.add(ops[op](a, b))
    if s.check() == z3.sat:
        m = s.model()
        model = {name: m[v].as_long() for name, v in vs.items()}
        return {"verdict": "accepted", "backend": "z3", "status": "consistent",
                "reasons": ["z3: constraints satisfiable"], "model": model}
    return {"verdict": "rejected", "backend": "z3", "status": "contradiction",
            "reasons": ["z3: constraints unsatisfiable (lattice violation)"], "model": None}


def _z3_contradiction(claims) -> dict:
    import z3

    s = z3.Solver()
    atoms: dict = {}
    for c in claims:
        key = (c.get("subject"), c.get("predicate"), c.get("object"))
        atoms.setdefault(key, z3.Bool("__".join(str(k) for k in key)))
    for c in claims:
        key = (c.get("subject"), c.get("predicate"), c.get("object"))
        lit = atoms[key]
        s.add(z3.Not(lit) if c.get("negated") else lit)
    if s.check() == z3.sat:
        return {"verdict": "accepted", "backend": "z3", "status": "consistent",
                "reasons": ["z3: no contradiction among asserted claims"], "model": None}
    return {"verdict": "rejected", "backend": "z3", "status": "contradiction",
            "reasons": ["z3: claim set is contradictory"], "model": None}


# ----------------------------------------------------------------------------- #
# Pure-Python fallback (exact for these decidable fragments)
# ----------------------------------------------------------------------------- #
def _fallback_lattice(assignments, constraints) -> dict:
    ops = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b,
           ">=": lambda a, b: a >= b, ">": lambda a, b: a > b,
           "==": lambda a, b: a == b, "!=": lambda a, b: a != b}
    for lhs, op, rhs in constraints:
        if op not in ops:
            return _err(f"unknown operator {op!r}")
        a = assignments.get(lhs, _as_int(lhs))
        b = assignments.get(rhs, _as_int(rhs))
        if a is None or b is None:
            return _err(f"unbound variable in constraint ({lhs} {op} {rhs})")
        if not ops[op](a, b):
            return {"verdict": "rejected", "backend": "fallback", "status": "contradiction",
                    "reasons": [f"violated: {lhs}({a}) {op} {rhs}({b})"], "model": None}
    return {"verdict": "accepted", "backend": "fallback", "status": "consistent",
            "reasons": ["fallback: all constraints hold"], "model": dict(assignments)}


def _fallback_contradiction(claims) -> dict:
    pos: set = set()
    neg: set = set()
    for c in claims:
        key = (c.get("subject"), c.get("predicate"), c.get("object"))
        (neg if c.get("negated") else pos).add(key)
    clash = pos & neg
    if clash:
        return {"verdict": "rejected", "backend": "fallback", "status": "contradiction",
                "reasons": [f"contradiction on {sorted(str(k) for k in clash)}"], "model": None}
    return {"verdict": "accepted", "backend": "fallback", "status": "consistent",
            "reasons": ["fallback: no contradictory atoms"], "model": None}


def _as_int(x: "Any") -> "int | None":
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _err(msg: str) -> dict:
    return {"verdict": "held", "backend": "fallback", "status": "error",
            "reasons": [msg], "model": None}


__all__ = [
    "z3_available",
    "check_lattice_consistency",
    "check_no_contradiction",
    "require_z3",
]
