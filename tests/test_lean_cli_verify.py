#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the trace()-free Lean verification path (agent.lean_backend.verify_lean_source).

This is the **L0 bypass**: lean-dojo's `trace()` deadlocks (macOS-arm64 AND Linux/CI —
see docs/06-Roadmap/Lean-L0-Trace-Deadlock.md), so `verify_lean_source` skips the tracer
and elaborates a self-contained source with the real `lean` CLI directly.

Two regimes:
  * FAIL-CLOSED (always runs, no Lean toolchain): verify_lean_source abstains with
    `lean_unavailable` — never crashes, never fabricates a verdict. This is the CI default.
  * REAL LEAN (runs only when the `lean` CLI is on PATH — e.g. the lean-kernel CI lane):
    a correct prelude-only proof -> accepted; a wrong proof -> rejected; a `sorry`/`admit`
    proof -> rejected (incomplete is NEVER accepted). This is a genuine L0 demonstration —
    a real Lean-kernel green-on-valid AND reject-on-invalid — with no deadlocking trace.

candidateOnly: a passing real-Lean run is L0 evidence (the formal-proof path runs once,
honestly), NOT a capability/AGI claim. Skipped (not failed) when Lean is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend  # noqa: E402


# --------------------------------------------------------------------------- #
# Fail-closed regime (always runs; no Lean toolchain needed)
# --------------------------------------------------------------------------- #


def test_verify_lean_source_abstains_without_lean() -> None:
    """No `lean` CLI -> abstain with lean_unavailable, never a fabricated verdict."""
    if lean_backend.lean_cli_available():
        return  # Lean present; the no-lean path is exercised by the real-Lean tests below
    r = lean_backend.verify_lean_source("theorem t : True := by trivial")
    assert r.verdict == "abstain", f"must abstain without lean, got {r.verdict}"
    assert "lean_unavailable" in r.reason
    assert r.to_dict()["detail"]["backend"] == "lean4-cli"


def test_verify_lean_source_backend_label() -> None:
    """The backend is labelled distinctly from the lean-dojo path (audit trail)."""
    r = lean_backend.verify_lean_source("theorem t : True := by trivial")
    assert r.backend == "lean4-cli"


# --------------------------------------------------------------------------- #
# Real Lean regime (only when the `lean` CLI is installed) — the L0 demonstration
# --------------------------------------------------------------------------- #


def _skip_if_no_lean():
    if not lean_backend.lean_cli_available():
        import pytest
        pytest.skip("`lean` CLI not on PATH; fail-closed path covered above")


def test_valid_proof_accepted() -> None:
    """L0 green-on-valid: a true prelude-only theorem with a correct proof -> accepted."""
    _skip_if_no_lean()
    r = lean_backend.verify_lean_source("theorem l0_valid : 1 + 1 = 2 := by rfl")
    assert r.verdict == "accepted", f"valid proof should be accepted: {r.reason}"
    assert r.goal_closed is True


def test_wrong_proof_rejected() -> None:
    """L0 reject-on-invalid: a false goal whose proof fails -> rejected, never accepted."""
    _skip_if_no_lean()
    r = lean_backend.verify_lean_source("theorem l0_bad : 1 + 1 = 3 := by rfl")
    assert r.verdict == "rejected", f"wrong proof must be rejected: {r.reason}"
    assert r.verdict != "accepted"


def test_sorry_is_not_accepted() -> None:
    """An incomplete (`sorry`) proof must be rejected — exit 0 + a warning is NOT a proof."""
    _skip_if_no_lean()
    r = lean_backend.verify_lean_source("theorem l0_sorry : 1 + 1 = 3 := by sorry")
    assert r.verdict == "rejected", f"sorry must not be accepted: {r.reason}"
    assert r.detail.get("sorry") is True


if __name__ == "__main__":
    # Script lane (repo convention: `python tests/test_X.py`). Each test guards on
    # `lean` availability: the fail-closed checks run when `lean` is ABSENT, the
    # real-kernel checks run when `lean` is on PATH.
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
