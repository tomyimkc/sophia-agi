#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Lean proof-search path (Path B).

Two regimes, both honest:
  * FAIL-CLOSED (always runs, no Lean): lean_backend + proof_search abstain when
    lean-dojo isn't installed. The CI default never breaks.
  * SEARCH LOGIC (runs with an injected scripted applier — no Lean needed): the
    best-first search finds a proof, the novelty probe fires, math_verifier's Lean
    delegation abstains cleanly. This proves the loop structure without a Lean toolchain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend, proof_search, math_verifier  # noqa: E402


def test_lean_backend_abstains_without_lean() -> None:
    """No lean-dojo installed (CI default) -> verify_proof abstains, never lies."""
    if lean_backend.lean_available():
        return  # lean-dojo present in this env; fail-closed path not exercised
    check = lean_backend.verify_proof(theorem="theorem t : True := by trivial", proof="trivial")
    assert check.verdict == "abstain"
    assert "lean_unavailable" in check.reason
    assert check.to_dict()["detail"]["lean"] is False


def test_math_verifier_lean_delegation_abstains_without_lean() -> None:
    """math_verifier.verify(use_lean=True) abstains identically when Lean is absent —
    the pre-wiring behavior is preserved exactly."""
    if lean_backend.lean_available():
        return
    r = math_verifier.verify("x", "theorem t : True", use_lean=True, lean_proof="trivial")
    assert r["verdict"] == "abstain"
    assert r["detail"]["backend"] == "lean"
    # and use_lean=True WITHOUT a proof also abstains cleanly
    r2 = math_verifier.verify("x", "theorem t : True", use_lean=True)
    assert r2["verdict"] == "abstain"


def test_novelty_probe_strict() -> None:
    """Strict novelty probe: a proof near-duplicate of corpus -> novel=False; a
    distinct proof -> novel=True. Per the human's 'strict' decision."""
    corpus = ["intro x; apply hx; exact x"]
    # near-duplicate (high trigram overlap) -> NOT novel
    near = "intro x; apply hx; exact x"
    n1 = lean_backend.novelty_check(near, corpus=corpus)
    assert n1["novel"] is False and n1["best_overlap"] > 0.9
    # genuinely distinct -> novel
    distinct = "induction n with | zero => rfl | succ n ih => rw [Nat.add_succ, ih]"
    n2 = lean_backend.novelty_check(distinct, corpus=corpus)
    assert n2["novel"] is True and n2["best_overlap"] < 0.5


def test_proof_search_finds_proof_with_injected_applier() -> None:
    """The search loop finds a proof when the injected tactic applier closes the goal.
    No Lean needed — this proves the search STRUCTURE (best-first + priority + budget)."""

    # Scripted applier: 'rfl' closes the goal; anything else is a no-op step.
    def apply(state, tactic):
        if tactic == "rfl":
            return "no goals", True
        return state + " (step)", False

    def proposer(theorem, state):
        return ["rfl", "intro x", "apply h"]  # 'rfl' is the winning tactic

    res = proof_search.search_proof(
        "theorem t : 0 = 0",
        proposer=proposer,
        initial_state="0 = 0",
        apply_tactic=apply,
        max_nodes=20,
        novelty_corpus=["intro x; apply hx"],
    )
    assert res.verdict == "proved"
    # injected-applier path: the applier IS the verifier, so a closed goal is proved
    # by construction; lean_verdict marks this honestly (no Lean was involved).
    assert res.lean_verdict == "accepted-by-injected-applier"
    assert "rfl" in (res.proof or "")
    assert res.novelty is not None  # novelty probe ran


def test_proof_search_abstains_within_budget() -> None:
    """No winning tactic within the node budget -> fail-closed 'no_proof_within_budget',
    never asserts an unproved goal."""

    def apply(state, tactic):
        return state + " step", False  # never closes

    def proposer(theorem, state):
        return ["intro x", "apply h"]

    res = proof_search.search_proof(
        "theorem t : False",
        proposer=proposer,
        initial_state="False",
        apply_tactic=apply,
        max_nodes=5,
    )
    assert res.verdict == "no_proof_within_budget"
    assert res.proof is None


def test_proof_search_report_discipline_fields() -> None:
    """No-overclaim: the report carries candidateOnly + level3Evidence: false."""
    res = proof_search.ProofSearchResult(verdict="no_proof_within_budget")
    d = res.to_dict()
    assert d["candidateOnly"] is True
    assert d["level3Evidence"] is False


def main() -> int:
    test_lean_backend_abstains_without_lean()
    test_math_verifier_lean_delegation_abstains_without_lean()
    test_novelty_probe_strict()
    test_proof_search_finds_proof_with_injected_applier()
    test_proof_search_abstains_within_budget()
    test_proof_search_report_discipline_fields()
    print("test_proof_search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
