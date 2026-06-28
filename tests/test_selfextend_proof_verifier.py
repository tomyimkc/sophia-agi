#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for selfextend/proof_verifier.py — kernel-as-oracle bridge for the flywheel.

Falsifiable invariants (the contract this module defends):
  1. Fail-closed reward: with the Lean kernel absent, kernel_verifier returns False for
     EVERY attempt, so proof_reward is 0.0 — no kernel, no reward. A missing toolchain
     can never be smuggled into a 1.0 reward.
  2. Fail-closed loop: with the kernel absent, close_loop_on_proofs returns
     loop_closed=False, routeAfter="abstain", and the report's reason names the
     wisdom-before-intelligence default. The loop NEVER promotes without a kernel.
  3. Anti-gaming is structurally near-vacuous for a kernel oracle: the train/held-out
     drop is 0.0 by construction (same oracle), and the report says so plainly.
  4. With a real kernel, a trivial-True attempt earns reward 1.0 and the loop can close
     (ADDITIONAL coverage; skipped, never failed, when Lean is absent).

These run green offline WITHOUT Lean (the fail-closed path IS the test).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent.lean_verifier as lv  # noqa: E402
import selfextend.proof_verifier as pv  # noqa: E402
from selfextend.proof_verifier import (  # noqa: E402
    ProofAttempt,
    close_loop_on_proofs,
    kernel_reward_is_hackable,
    kernel_verifier,
    mean_proof_reward,
    proof_reward,
)


def _attempts() -> "list[ProofAttempt]":
    return [
        ProofAttempt("add_comm", "∀ n m : Nat, n + m = m + n", "by omega"),
        ProofAttempt("true", "True", "trivial"),
        ProofAttempt("sorry_bad", "False", "by sorry"),  # a non-proof
    ]


# --------------------------------------------------------------------------- #
# 1. Fail-closed reward without a kernel
# --------------------------------------------------------------------------- #


def test_kernel_verifier_false_without_kernel(monkeypatch) -> None:
    monkeypatch.setattr(lv, "lean_available", lambda: False)
    for a in _attempts():
        assert kernel_verifier(a) is False, f"{a.claim_id} accepted without kernel"


def test_proof_reward_zero_without_kernel(monkeypatch) -> None:
    monkeypatch.setattr(lv, "lean_available", lambda: False)
    assert all(proof_reward(a) == 0.0 for a in _attempts())
    assert mean_proof_reward(_attempts()) == 0.0


def test_kernel_verifier_rejects_non_attempt(monkeypatch) -> None:
    """Adversarial: garbage input must not crash and must not earn reward."""
    monkeypatch.setattr(lv, "lean_available", lambda: False)
    assert kernel_verifier("not an attempt") is False
    assert kernel_verifier(None) is False


# --------------------------------------------------------------------------- #
# 2. Fail-closed loop without a kernel
# --------------------------------------------------------------------------- #


def test_close_loop_abstains_without_kernel(monkeypatch) -> None:
    # Patch lean_available at its CALL SITE (selfextend.proof_verifier), not at its
    # definition site (agent.lean_verifier). proof_verifier binds `lean_available`
    # into its own namespace via `from agent.lean_verifier import ... lean_available`,
    # so patching lv.lean_available has no effect on the binding pv.close_loop_on_proofs
    # actually consults. (This was latent — it only passed pre-Lean because both refs
    # returned False; the lean-kernel CI lane caught it once Lean was actually present.)
    monkeypatch.setattr(pv, "lean_available", lambda: False)
    r = close_loop_on_proofs("nat-arithmetic", _attempts(), _attempts(), _attempts())
    assert r["loop_closed"] is False
    assert r["promoted"] is False
    assert r["routeAfter"] == "abstain"
    assert r["heldoutReward"] == 0.0
    assert r["invariants"]["kernel_present"] is False
    assert r["invariants"]["competence_flips_abstain_to_answer"] is False
    # The reason must name the wisdom-before-intelligence default, not "error".
    assert "abstained" in r["reason"] or "abstain" in r["reason"].lower()


def test_close_loop_promotion_below_threshold_abstains(monkeypatch) -> None:
    """With a kernel present but a low held-out reward, the loop still abstains —
    promotion needs the policy to actually clear the bar."""
    monkeypatch.setattr(pv, "lean_available", lambda: True)
    # Force the kernel backend to reject every attempt (simulate "no proofs found").
    # Patch at the call site (pv), for the same reason as above.
    monkeypatch.setattr(pv, "check_proof",
                        lambda *a, **k: {"verdict": "held", "status": "unprovable_here"})
    r = close_loop_on_proofs("hard-domain", _attempts(), _attempts(), _attempts(),
                             threshold=1.0)
    assert r["loop_closed"] is False
    assert r["promoted"] is False
    assert r["routeAfter"] == "abstain"


# --------------------------------------------------------------------------- #
# 3. Anti-gaming is structurally near-vacuous for a kernel oracle
# --------------------------------------------------------------------------- #


def test_anti_gaming_drop_is_zero_for_kernel_oracle(monkeypatch) -> None:
    monkeypatch.setattr(lv, "lean_available", lambda: False)
    rep = kernel_reward_is_hackable(_attempts())
    # Same oracle for train and held-out -> drop is identically 0.0 -> not hacked.
    assert rep["drop"] == 0.0
    assert rep["hacked"] is False
    assert "structurally" in rep["interpretation"].lower()


# --------------------------------------------------------------------------- #
# 4. Real-kernel path (additional; skipped when Lean absent)
# --------------------------------------------------------------------------- #


def test_real_kernel_closes_loop_on_trivial() -> None:
    """If Lean is installed, a trivial-True attempt earns reward 1.0 and the loop closes
    on a trivial domain. ADDITIONAL coverage — skipped, never failed, when absent."""
    if not lv.lean_available():
        import pytest
        pytest.skip("Lean toolchain not installed; offline fail-closed path covered above")
    a = ProofAttempt("trivial_true", "True", "trivial")
    assert proof_reward(a) == 1.0, "trivial proof of True must earn reward 1.0 under a real kernel"
    r = close_loop_on_proofs("trivial", [a], [a], [a], threshold=1.0)
    assert r["loop_closed"] is True
    assert r["routeAfter"] == "answer"


def main() -> int:
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
