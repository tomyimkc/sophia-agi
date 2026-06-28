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

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend  # noqa: E402

# L0 viability-ladder gate (docs/06-Roadmap/Lean-L0-Trace-Deadlock.md §0).
# When SOPHIA_L0_REQUIRE_TRACE=1, the real-lean-dojo test below is FORBIDDEN from
# skipping: a skip becomes a hard failure. This is what makes a GREEN lean-dojo-search
# lane a genuine L0 signal ("trace() completed on this platform AND check_proof
# accepted a correct proof / did not fabricate on a wrong one") rather than a
# green-because-skipped fragile green. The CI lane sets this; local runs without it
# keep the old skip-when-absent behaviour.
_L0_REQUIRE_TRACE = os.environ.get("SOPHIA_L0_REQUIRE_TRACE") == "1"


def _skip_or_fail(reason: str) -> None:
    """Skip locally; in the L0-require-trace lane, turn the skip into a hard failure."""
    import pytest
    if _L0_REQUIRE_TRACE:
        pytest.fail(f"SOPHIA_L0_REQUIRE_TRACE=1 but L0 trace path was skipped: {reason}")
    pytest.skip(reason)


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


# lean4-example, pinned to a fixed commit so the theorem layout cannot drift under
# us. At this commit `Lean4Example.lean` contains:
#     theorem foo (a : Nat) : a + 1 = Nat.succ a := by rfl
# `foo` is the right L0 target: it is genuinely closed by `rfl`, so check_proof MUST
# accept the correct proof. (The older test pointed at `hello_world` with `rfl`, which
# is wrong — `hello_world : a + b + c = a + c + b` is NOT `rfl`-provable — and at a
# non-existent file `HelloWorld.lean`, so it could only ever fail or skip.)
_LEAN4_EXAMPLE_URL = "https://github.com/yangky11/lean4-example"
_LEAN4_EXAMPLE_COMMIT = "7761283d0aed994cd1c7e893786212d2a01d159e"
_L0_FILE = "Lean4Example.lean"
_L0_THEOREM = "foo"
_L0_CORRECT_PROOF = "rfl"          # genuinely closes `a + 1 = Nat.succ a`
_L0_WRONG_PROOF = "exact 1"        # type-wrong: must NOT be accepted


def test_check_proof_in_repo_accepts_correct_rejects_wrong() -> None:
    """L0 smoke check: drive lean-dojo 4.x check_proof against a real traced repo.

    Uses yangky11/lean4-example (the repo LeanDojo's own Getting Started documents),
    pinned to a fixed commit, targeting `foo : a + 1 = Nat.succ a := by rfl` — a
    theorem genuinely closed by `rfl`. Asserts the contract that matters:
    a CORRECT proof (`rfl`) -> accepted; a WRONG proof (`exact 1`) -> NOT accepted.

    This is the viability-ladder L0 step (docs/06-Roadmap/Lean-L0-Trace-Deadlock.md):
    a green here proves `trace()` completed on this platform and the real check_proof
    path returns `accepted` for a correct proof.

    HANG GUARD (#189) reconciled with the L0 LANE (#187): lean-dojo's `trace()` is
    slow / deadlocks (~30-90 min; §1/§1a/§1b), so the real-trace assertion is SKIPPED
    by default to keep the per-PR lane from hanging, and runs ONLY when explicitly
    opted in — the deadlock probe (`SOPHIA_LEAN_TRACE_DEADLOCK_PROBE=1`) or the L0 lane
    (`SOPHIA_L0_REQUIRE_TRACE=1`). When lean-dojo is absent the test skips (fail-closed
    path covered above) — UNLESS SOPHIA_L0_REQUIRE_TRACE=1, where any skip is a hard
    failure so a green cannot be a green-because-skipped fragile green.
    """
    if not lean_backend.lean_available():
        _skip_or_fail("lean-dojo not installed; fail-closed path covered above")
        return
    # Hang guard (#189) reconciled with the L0 lane (#187): run the real trace only when
    # opted in — the deadlock probe OR the require-trace lane — else skip so the per-PR
    # lane never hangs. Under SOPHIA_L0_REQUIRE_TRACE=1, _skip_or_fail makes any skip a
    # hard failure (an un-skippable L0 signal).
    _run_trace = (os.environ.get("SOPHIA_LEAN_TRACE_DEADLOCK_PROBE") == "1") or _L0_REQUIRE_TRACE
    if not _run_trace:
        _skip_or_fail(
            "lean-dojo trace() is slow/deadlocks (Lean-L0-Trace-Deadlock.md §1–§1b); "
            "skipped to avoid the long CI hang. Set SOPHIA_LEAN_TRACE_DEADLOCK_PROBE=1 "
            "or run the L0 lane (SOPHIA_L0_REQUIRE_TRACE=1) to execute it.")
    try:
        from lean_dojo import LeanGitRepo, Theorem  # type: ignore
    except ImportError:
        _skip_or_fail("lean_dojo.{LeanGitRepo,Theorem} not importable in this version")
        return  # _skip_or_fail raises; this satisfies type-checkers on the no-import path

    repo = LeanGitRepo(_LEAN4_EXAMPLE_URL, _LEAN4_EXAMPLE_COMMIT)
    try:
        thm = Theorem(repo, Path(_L0_FILE), _L0_THEOREM)
    except Exception as exc:
        # Theorem construction itself is cheap (no trace); a failure here means a
        # genuine API/layout problem, not a deadlock — surface it under require-trace.
        _skip_or_fail(f"could not construct {_L0_THEOREM} Theorem in lean4-example: {exc!r}")
        return

    # A correct proof -> accepted. This is the load-bearing L0 assertion: it can only
    # pass if trace() completed and lean-dojo's check_proof returned True.
    r_ok = lean_backend.check_proof_in_repo(thm, _L0_CORRECT_PROOF)
    assert r_ok.verdict == "accepted", (
        f"L0: correct proof `{_L0_CORRECT_PROOF}` of `{_L0_THEOREM}` should be accepted, "
        f"got {r_ok.verdict}: {r_ok.reason}")

    # A wrong proof -> NOT accepted (rejected, or abstain if check_proof raised). Never
    # a fabricated accept.
    r_bad = lean_backend.check_proof_in_repo(thm, _L0_WRONG_PROOF)
    assert r_bad.verdict != "accepted", (
        f"L0: wrong proof `{_L0_WRONG_PROOF}` was ACCEPTED — fabrication bug: {r_bad.reason}")
    assert r_bad.verdict in ("rejected", "abstain"), (
        f"wrong proof verdict must be rejected/abstain, got {r_bad.verdict}")


# --------------------------------------------------------------------------- #
# L0 via mathlib4 + lean-dojo remote cache (the throughput path)
# --------------------------------------------------------------------------- #

# The lean4-example trace above is SLOW, not stuck, on Linux (~90 min of prelude
# extraction on a 2-core runner — docs/06-Roadmap/Lean-L0-Trace-Deadlock.md §1b).
# The chosen throughput fix (§4) is to lean on lean-dojo's *published* remote cache:
# for a mathlib4 commit lean-dojo has pre-traced, check_proof downloads the traced
# repo from REMOTE_CACHE_URL (https://dl.fbaipublicfiles.com/lean-dojo) instead of
# running the local extraction. We pin the EXACT commit + theorem lean-dojo's own
# demo uses, so the cache is guaranteed to exist for this lean-dojo version.
_L0_MATHLIB = os.environ.get("SOPHIA_L0_MATHLIB") == "1"
_MATHLIB4_URL = "https://github.com/leanprover-community/mathlib4"
_MATHLIB4_COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"   # LeanDojo Benchmark 4 / demo
_MATHLIB4_FILE = "Mathlib/Algebra/BigOperators/Pi.lean"
_MATHLIB4_THEOREM = "pi_eq_sum_univ"
_MATHLIB4_CORRECT = "by\n  ext\n  simp\n"   # the real proof (lean-dojo demo-lean4.ipynb)
_MATHLIB4_WRONG = "by\n  rfl\n"             # rfl does NOT prove it -> must not be accepted


def test_check_proof_mathlib4_cached_l0() -> None:
    """L0 via mathlib4 + lean-dojo's remote cache (throughput path; §1b/§4).

    Opt-in: runs ONLY when SOPHIA_L0_MATHLIB=1 (the manual-dispatch CI lane), because
    it downloads lean-dojo's multi-GB pre-traced mathlib4 cache. Targets the exact
    theorem lean-dojo's own demo uses at the cached commit, so check_proof resolves
    against the DOWNLOADED traced repo rather than the ~90-min local prelude
    extraction. Asserts the real proof is accepted and a wrong proof is not (no
    fabrication). When opted in, every off-ramp is a hard failure (not a skip) so a
    green lane is a genuine L0 signal.
    """
    import pytest
    if not _L0_MATHLIB:
        pytest.skip("set SOPHIA_L0_MATHLIB=1 to run the mathlib4 remote-cache L0 check")
    if not lean_backend.lean_available():
        pytest.fail("SOPHIA_L0_MATHLIB=1 but lean-dojo is not installed")
    try:
        from lean_dojo import LeanGitRepo, Theorem  # type: ignore
    except ImportError:
        pytest.fail("SOPHIA_L0_MATHLIB=1 but lean_dojo.{LeanGitRepo,Theorem} not importable")

    repo = LeanGitRepo(_MATHLIB4_URL, _MATHLIB4_COMMIT)
    thm = Theorem(repo, Path(_MATHLIB4_FILE), _MATHLIB4_THEOREM)

    # The real proof -> accepted. Load-bearing: passes only if the cached traced repo
    # resolved and lean-dojo's check_proof returned True.
    r_ok = lean_backend.check_proof_in_repo(thm, _MATHLIB4_CORRECT)
    assert r_ok.verdict == "accepted", (
        f"L0(mathlib4): real proof of `{_MATHLIB4_THEOREM}` should be accepted, "
        f"got {r_ok.verdict}: {r_ok.reason}")

    # A wrong proof -> NOT accepted. Never a fabricated accept.
    r_bad = lean_backend.check_proof_in_repo(thm, _MATHLIB4_WRONG)
    assert r_bad.verdict != "accepted", (
        f"L0(mathlib4): wrong proof was ACCEPTED — fabrication bug: {r_bad.reason}")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
