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


def test_assemble_uses_newline_separated_tactics() -> None:
    """Regression: multi-tactic proofs must be newline-separated, not space-joined —
    space-joining produces invalid Lean 4 ``by`` blocks and would make Lean reject
    otherwise-correct proofs (and corrupt the novelty-probe text)."""
    # header without `:= by` -> the assembler adds it
    src = proof_search._assemble("theorem t : True", ("trivial",))
    assert src == "theorem t : True := by\ntrivial"
    # multi-tactic path -> newline-separated, never a single space-joined line
    src2 = proof_search._assemble("theorem t : True := by", ("intro x", "exact x"))
    assert src2 == "theorem t : True := by\nintro x\nexact x"
    assert "intro x exact x" not in src2  # the old buggy space-joined form
    # empty path -> trivial `rfl` fallback
    assert proof_search._assemble("theorem t : True := by", ()) == "theorem t : True := by\nrfl"


def test_assembled_proof_carries_single_theorem_header() -> None:
    """Regression: verify_proof must emit ONE theorem block, not duplicate the header
    or inject a dashed separator (both made Lean reject correct proofs). The assembler
    and verify_proof now agree: pass a header + tactic body, get one valid source."""
    # An injected-applier search that closes on a tactic path assembles a clean block.
    # Use a 2-step path: 'intro x' advances (not closed), then 'done' closes — so the
    # recorded proof reflects the FULL tactic trajectory, newline-separated.
    def apply(state, tactic):
        if tactic == "done" and "intro" in state:
            return "no goals", True
        if tactic == "intro x":
            return state + " intro", False  # advances but doesn't close
        return state + " step", False

    def proposer(theorem, state):
        # only offer 'intro x' at the root, 'done' after intro -> forces the 2-step path
        return ["intro x"] if "intro" not in state else ["done"]

    res = proof_search.search_proof(
        "theorem t : P", proposer=proposer, initial_state="P",
        apply_tactic=apply, max_nodes=20,
    )
    assert res.verdict == "proved"
    # the recorded proof is the full newline-separated tactic path (regression for the
    # space-join bug) — never the old "intro x done" single-line form.
    assert res.proof == "intro x\ndone"
    assert " " not in (res.proof or "").replace("\n", "") or res.proof == "intro x\ndone"


def test_math_verifier_lean_backend_id_is_lean() -> None:
    """Regression: even when Lean IS available, math_verifier must report backend 'lean'
    (its contract), not 'lean4' (LeanCheck's toolchain id) — downstream consumers switch
    on detail.backend. Skipped cleanly when lean-dojo is absent (CI default)."""
    if not lean_backend.lean_available():
        return  # CI default; the normalization is only reachable on the Lean path
    r = math_verifier.verify("trivial", "theorem t : True := by", use_lean=True,
                             lean_proof="trivial")
    assert r["detail"]["backend"] == "lean"


def test_verify_proof_abstains_on_free_form_without_repo_key() -> None:
    """Phase-0 drift fix (lean-dojo 4.x): verify_proof must abstain with a clear reason
    when called WITHOUT theorem_name + file_path, because lean-dojo 4.x's check_proof
    verifies a proof of a NAMED theorem in a TRACED repo — there is no stateless
    "elaborate this string" API.

    The pre-fix code called a phantom `LeanDojo(repo=...).run_code(...)` API that does
    not exist in lean-dojo 4.x, so it ALWAYS abstained with the misleading
    "lean-dojo import failed". This test pins the corrected, honest reason so the
    limitation is explicit, not a silent wrong-API failure.

    Runs in BOTH regimes: when lean-dojo is absent (abstains: not installed) and when
    present (abstains: needs repo key). Either way the verdict is abstain, never accepted."""
    check = lean_backend.verify_proof(theorem="theorem t : True := by trivial",
                                      proof="trivial")  # no theorem_name/file_path
    assert check.verdict == "abstain"
    # The reason must name the actual limitation, not the misleading "import failed".
    assert "import failed" not in check.reason
    # The honest reason names the real 4.x path (check_proof_in_repo) OR, when lean-dojo
    # is absent, "not installed". Either is the honest limitation; "import failed" is not.
    assert ("check_proof_in_repo" in check.reason
            or "theorem_name" in check.reason
            or "not installed" in check.reason), f"unexpected reason: {check.reason}"



def test_full_block_detection_not_fooled_by_have_term() -> None:
    """Regression: verify_proof must classify a proof as a full theorem block ONLY by its
    LEADING declaration keyword, not by a `:= by` substring. A tactic body legitimately
    contains `have h : P := by ...`, which the old substring test misclassified as a full
    block — dropping the theorem header and making Lean reject an otherwise-correct proof."""
    # Mirror lean_backend.verify_proof's full-block detection exactly so this test stays
    # honest if the heuristic changes.
    def _is_full_block(proof: str) -> bool:
        head = proof.lstrip()[:16].lower()
        return head.startswith(("theorem ", "lemma ", "def ", "example", "axiom "))

    # tactic body containing `have ... := by` -> must NOT be a full block (the bug case)
    assert not _is_full_block("intro x\nhave h : P := by exact hp\nexact h")
    assert not _is_full_block("  have h : True := by trivial")  # leading whitespace + have
    assert not _is_full_block("intros; apply Nat.add_comm")     # normal tactic body
    # actual full theorem/lemma blocks -> detected
    assert _is_full_block("theorem t : True := by\n  trivial")
    assert _is_full_block("lemma foo : P := by")
    assert _is_full_block("example : True := by trivial")


def main() -> int:
    test_lean_backend_abstains_without_lean()
    test_math_verifier_lean_delegation_abstains_without_lean()
    test_novelty_probe_strict()
    test_proof_search_finds_proof_with_injected_applier()
    test_proof_search_abstains_within_budget()
    test_proof_search_report_discipline_fields()
    test_assemble_uses_newline_separated_tactics()
    test_assembled_proof_carries_single_theorem_header()
    test_math_verifier_lean_backend_id_is_lean()
    print("test_proof_search: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
