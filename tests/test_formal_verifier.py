#!/usr/bin/env python3
"""Tests for agent/formal_verifier.py — optional-Z3 formal verifier (Stage C).

Falsifiable invariants:
  1. Lattice consistency: a derived label dominating its parent is accepted;
     a write-down (derived < parent) is rejected.
  2. Contradiction detection: a positive + negative co-assertion is rejected;
     a consistent set is accepted.
  3. Fail-closed: require_z3 returns held/z3_unavailable when z3 is absent — it
     NEVER returns accepted without a real solver.
  4. The fallback is exact for these decidable fragments (same verdict z3 would
     give), so CI is green offline without z3.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.formal_verifier import (  # noqa: E402
    check_lattice_consistency,
    check_no_contradiction,
    require_z3,
    z3_available,
)


def test_lattice_dominance_accepted() -> None:
    r = check_lattice_consistency({"derived": 2, "parent": 0}, [("derived", ">=", "parent")])
    assert r["verdict"] == "accepted"
    assert r["status"] == "consistent"


def test_lattice_write_down_rejected() -> None:
    r = check_lattice_consistency({"derived": 0, "parent": 2}, [("derived", ">=", "parent")])
    assert r["verdict"] == "rejected"
    assert r["status"] == "contradiction"


def test_lattice_unbound_variable_held() -> None:
    # only meaningful for the fallback path (z3 would treat as IntVal); guard it
    if not z3_available():
        r = check_lattice_consistency({"a": 1}, [("a", "<=", "ghost_var_xyz")])
        assert r["verdict"] == "held"
        assert r["status"] == "error"


def test_contradiction_rejected() -> None:
    claims = [
        {"subject": "confucius", "predicate": "authored", "object": "dao_de_jing", "negated": False},
        {"subject": "confucius", "predicate": "authored", "object": "dao_de_jing", "negated": True},
    ]
    r = check_no_contradiction(claims)
    assert r["verdict"] == "rejected"
    assert r["status"] == "contradiction"


def test_consistent_claims_accepted() -> None:
    claims = [
        {"subject": "laozi", "predicate": "authored", "object": "dao_de_jing", "negated": False},
        {"subject": "confucius", "predicate": "authored", "object": "analects", "negated": False},
    ]
    r = check_no_contradiction(claims)
    assert r["verdict"] == "accepted"
    assert r["status"] == "consistent"


def test_require_z3_is_failclosed_when_absent() -> None:
    r = require_z3(check_no_contradiction, [])
    if z3_available():
        assert r["verdict"] in ("accepted", "rejected")
    else:
        assert r["verdict"] == "held"
        assert r["status"] == "z3_unavailable"
        # the critical safety property: never silently accepted
        assert r["verdict"] != "accepted"


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_formal_verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
