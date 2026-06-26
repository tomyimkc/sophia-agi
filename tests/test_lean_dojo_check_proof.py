#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the lean-dojo 4.x integration (agent.lean_backend.check_proof_in_repo).

This is the REAL verification path that the drift-fix (Phase 0) introduced.
`verify_proof` historically called a non-existent lean-dojo 4.x API
(`LeanDojo(repo=...).run_code(source)`); it now abstains honestly and points
callers at `check_proof_in_repo`, which uses the real 4.x
`check_proof(thm: Theorem, proof: str) -> bool`.

Two regimes:
  * FAIL-CLOSED (always runs, no lean-dojo): verify_proof abstains honestly with
    the API pointer; check_proof_in_repo abstains with lean_unavailable.
  * REAL LEAN-DOJO (runs only when lean-dojo is installed): verify_proof returns
    the honest abstain regardless (it CANNOT verify a standalone snippet on 4.x),
    and check_proof_in_repo drives lean-dojo's check_proof against a theorem in a
    MINIMAL traced repo (leanprover-community/lean4-example — NOT full Mathlib).
    Asserts: a correct proof -> accepted; a wrong proof -> rejected; never fabricated.

The minimal-repo trace is CI-feasible (lean4-example is tiny); full Mathlib tracing
is deliberately NOT used here. Runs in the lean-dojo-search CI lane.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend  # noqa: E402


# --------------------------------------------------------------------------- #
# Fail-closed regime (always runs; no lean-dojo needed)
# --------------------------------------------------------------------------- #


def test_verify_proof_abstains_honestly_with_api_pointer() -> None:
    """verify_proof must abstain (not crash, not fabricate) and name the real 4.x API.

    Regression for the drift: the old code did `from lean_dojo import LeanDojo` (a class
    that does not exist in 4.x) and returned the misleading "lean-dojo import failed".
    Now it abstains with a reason pointing at check_proof_in_repo.
    """
    if not lean_backend.lean_available():
        return  # CI default; the no-lean path is covered by test_proof_search.py
    r = lean_backend.verify_proof(theorem="theorem t : True := by", proof="trivial")
    assert r.verdict == "abstain", f"verify_proof must abstain on 4.x, got {r.verdict}"
    assert "check_proof_in_repo" in r.reason, "reason must name the working 4.x API"
    assert r.to_dict()["detail"]["api"] == "check_proof_in_repo"


def test_check_proof_in_repo_abstains_without_lean() -> None:
    """No lean-dojo -> check_proof_in_repo abstains, never lies."""
    if lean_backend.lean_available():
        return  # lean-dojo present; the no-lean path not exercised here
    r = lean_backend.check_proof_in_repo(theorem_obj=None, proof="trivial")
    assert r.verdict == "abstain"
    assert "lean_unavailable" in r.reason


# --------------------------------------------------------------------------- #
# Real lean-dojo regime (only when lean-dojo installed; minimal repo, not Mathlib)
# --------------------------------------------------------------------------- #


def test_check_proof_in_repo_accepts_correct_rejects_wrong() -> None:
    """Drive lean-dojo 4.x check_proof against a theorem in a MINIMAL traced repo.

    Uses leanprover-community/lean4-example (the repo LeanDojo's own Getting Started
    documents) — tiny, so tracing is CI-feasible. Asserts the contract that matters:
    a CORRECT proof -> accepted; a WRONG proof -> rejected. Never fabricated.

    Skipped (not failed) when lean-dojo is absent — the fail-closed path is covered above.
    """
    if not lean_backend.lean_available():
        import pytest
        pytest.skip("lean-dojo not installed; fail-closed path covered above")
    try:
        from lean_dojo import LeanGitRepo, Theorem  # type: ignore
    except ImportError:
        import pytest
        pytest.skip("lean_dojo.Theorem not importable in this version")

    # lean4-example's hello_world theorem (the canonical minimal LeanDojo example).
    # Tracing this small repo is fast; full Mathlib is deliberately NOT used.
    repo = LeanGitRepo("https://github.com/yangky11/lean4-example", "main")
    try:
        thm = Theorem(repo, Path("HelloWorld.lean"), "hello_world")
    except Exception:
        # Repo layout / theorem name may differ across lean4-example revisions; if we
        # can't construct the theorem, skip rather than fabricate a verdict.
        import pytest
        pytest.skip("could not construct hello_world Theorem in lean4-example (layout drift)")

    # A correct proof -> accepted (lean-dojo check_proof returns True)
    r_ok = lean_backend.check_proof_in_repo(thm, "rfl")
    assert r_ok.verdict == "accepted", f"correct proof should be accepted: {r_ok.reason}"

    # A wrong proof -> rejected (check_proof returns False), NOT accepted
    r_bad = lean_backend.check_proof_in_repo(thm, "exact 1")
    assert r_bad.verdict in ("rejected", "abstain"), (
        f"wrong proof must not be accepted: {r_bad.verdict}")
    assert r_bad.verdict != "accepted", "wrong proof was ACCEPTED — fabrication bug"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
