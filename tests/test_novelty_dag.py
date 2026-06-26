#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the tactic-DAG novelty hash (§3.3 / §5.3 of the Open-Problems critique).

The DAG hash is the STRICT decider that char-trigram Jaccard cannot be: it detects a
re-proof via a *different tactic path* as a duplicate of the structural proof, while
correctly distinguishing two genuinely different tactic paths. These tests pin the
four load-bearing invariants from the spec:

  (a) a proof and its commutatively-reordered variant hash IDENTICALLY;
  (b) two genuinely different tactic paths to the same theorem hash DIFFERENTLY;
  (c) an unparseable proof ABSTAINS cleanly (dag_hash None, novel False, never lies);
  (d) the whole thing runs with stdlib only — no Lean, no model — so CI is unchanged.

Plus the discipline invariants that must NOT regress:
  * novelty_check(method="trigram") still returns the legacy dict shape (backward-compat);
  * novelty_check(method="auto") is the conjunction: passes trigram THEN the DAG decider;
  * no verdict ever promotes the ladder — every result is a measurement, not a claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend  # noqa: E402


# ---------------------------------------------------------------------------
# (a) Commutative / associative rewrite reordering -> IDENTICAL hash.
#     `rw [Nat.add_comm, Nat.mul_comm]` must hash the same as
#     `rw [Nat.mul_comm, Nat.add_comm]` (permutation of order-irrelevant rewrites).
# ---------------------------------------------------------------------------
def test_dag_commutative_reorder_hashes_identically() -> None:
    """A proof and its comm/assoc-rewrite-permuted variant share one DAG hash."""
    base = "rw [Nat.add_comm, Nat.mul_comm]; ring"
    reordered = "rw [Nat.mul_comm, Nat.add_comm]; ring"

    d_base = lean_backend.novelty_check_dag(base, corpus=[])
    d_re = lean_backend.novelty_check_dag(reordered, corpus=[])

    assert d_base["dag_parsed"] and d_re["dag_parsed"]
    assert d_base["dag_hash"] is not None
    assert d_base["dag_hash"] == d_re["dag_hash"], (
        f"comm/assoc reorder changed the DAG hash: {d_base['dag_hash']} != {d_re['dag_hash']}"
    )


def test_dag_non_commutative_rewrite_keeps_order() -> None:
    """Non-comm rewrite lemmas are order-LOAD-BEARING: their order must NOT collapse
    (collapsing it would over-claim novelty). Only comm/assoc lemmas permute."""
    seq_a = "rw [Nat.add_comm, le_add_iff_nonneg]; ring"
    seq_b = "rw [le_add_iff_nonneg, Nat.add_comm]; ring"  # non-comm in different slot

    d_a = lean_backend.novelty_check_dag(seq_a, corpus=[])
    d_b = lean_backend.novelty_check_dag(seq_b, corpus=[])
    assert d_a["dag_hash"] != d_b["dag_hash"], "non-comm rewrite order collapsed (would over-claim)"


# ---------------------------------------------------------------------------
# (b) Genuinely different tactic paths -> DIFFERENT hashes.
#     Same theorem, but induction vs. a rewrite chain is a different proof structure.
# ---------------------------------------------------------------------------
def test_dag_distinct_paths_hash_differently() -> None:
    """Two different tactic paths to the same theorem get distinct DAG hashes."""
    # Path 1: induction over n.
    p1 = "induction n with | zero => rfl | succ n ih => rw [Nat.add_succ, ih]"
    # Path 2: a rewriter chain (no induction, no local hypothesis).
    p2 = "rw [Nat.add_comm, Nat.add_zero]"

    d1 = lean_backend.novelty_check_dag(p1, corpus=[])
    d2 = lean_backend.novelty_check_dag(p2, corpus=[])

    assert d1["dag_parsed"] and d2["dag_parsed"]
    assert d1["dag_hash"] != d2["dag_hash"], "distinct tactic paths hashed identically (under-detects)"
    # And the DAG faithfully captures the structural difference: p1 has an induction
    # local-hyp dependency edge (ih), p2 does not.
    dag1 = lean_backend._build_tactic_dag(p1)
    assert dag1["edges"], "induction path should have a local-hyp dependency edge"


def test_dag_corpus_duplicate_detected() -> None:
    """A corpus proof with the same normalized DAG -> novel=False (the point of the
    hash: stop a re-proof via a different-but-structurally-identical tactic path)."""
    corpus_proof = "induction n with | zero => rfl | succ n ih => rw [Nat.add_succ, ih]"
    # A DIFFERENT theorem name, fresh variable names — but the same tactic STRUCTURE
    # (induction -> rfl / rw[add_succ, ih]). The DAG normalizes the names away.
    same_structure = "induction k with | zero => rfl | succ k jk => rw [Nat.add_succ, jk]"
    d = lean_backend.novelty_check_dag(same_structure, corpus=[corpus_proof])
    assert d["dag_parsed"]
    assert d["novel"] is False, "structurally-identical tactic path scored novel (false positive)"


# ---------------------------------------------------------------------------
# (c) Unparseable / empty proof -> ABSTAIN cleanly. Never fabricate a hash.
# ---------------------------------------------------------------------------
def test_dag_abstains_on_unparseable() -> None:
    """Garbage / empty input: dag_hash is None, novel is False. Fail-closed —
    never fabricates a hash to claim novelty."""
    for bad in ["", "    ", "```", "this is not a proof at all"]:
        d = lean_backend.novelty_check_dag(bad, corpus=[])
        assert d["dag_hash"] is None, f"fabricated a hash on unparseable input: {bad!r}"
        assert d["dag_parsed"] is False
        assert d["novel"] is False, "claimed novelty on an unparseable proof (over-claim)"


def test_dag_abstains_on_comment_only() -> None:
    """A proof that is only comments / whitespace has no tactic -> abstains."""
    d = lean_backend.novelty_check_dag("-- nothing here\n  \n", corpus=[])
    assert d["dag_parsed"] is False and d["novel"] is False


# ---------------------------------------------------------------------------
# (d) Backward compatibility + the method= dispatch.
# ---------------------------------------------------------------------------
def test_novelty_check_trigram_method_legacy_shape() -> None:
    """method='trigram' returns the EXACT legacy dict shape (no dag_* keys) so
    pre-existing callers in proof_search.py and test_proof_search.py are untouched."""
    out = lean_backend.novelty_check(
        "intro x; apply hx; exact x", corpus=["intro x; apply hx; exact x"], method="trigram"
    )
    # legacy keys present
    assert set(["novel", "best_overlap", "threshold", "method", "proof_hash"]).issubset(out)
    # legacy key only ever had these top-level keys (no dag_* leakage)
    assert "dag_hash" not in out and "dag_parsed" not in out
    assert out["novel"] is False and out["best_overlap"] > 0.9


def test_novelty_check_default_is_auto_and_uses_dag() -> None:
    """The default method='auto' runs the trigram pre-filter then the DAG decider.
    On a near-duplicate it short-circuits; on a structurally-novel proof it applies DAG."""
    corpus = ["induction n with | zero => rfl | succ n ih => rw [Nat.add_succ, ih]"]

    # Near-duplicate (verbatim) -> auto short-circuits via trigram, novel=False.
    auto_dup = lean_backend.novelty_check(corpus[0], corpus=corpus)
    assert auto_dup["novel"] is False

    # A genuinely distinct proof -> trigram passes, DAG decider applied, novel=True.
    distinct = "simp [Nat.mul_comm]; ring"
    auto_novel = lean_backend.novelty_check(distinct, corpus=corpus)
    assert auto_novel["dag_decider"] == "applied"
    assert auto_novel["novel"] is True
    assert auto_novel["dag_hash"] is not None


def test_novelty_check_auto_does_not_overclaim_on_unparseable() -> None:
    """auto + unparseable proof: the DAG abstains, and 'auto' must NOT claim novelty
    just because trigram overlap is low. Low overlap ≠ novel."""
    out = lean_backend.novelty_check("garbage not a proof", corpus=["induction n; rfl"], method="auto")
    assert out["novel"] is False, "auto claimed novelty where the DAG could only abstain"


# ---------------------------------------------------------------------------
# Discipline invariant: no result is a capability claim. (Mirror of the
# proof_search discipline test — the novelty probe is a measurement only.)
# ---------------------------------------------------------------------------
def test_novelty_results_carry_no_capability_claim() -> None:
    """Every novelty result is a measurement dict; none carry candidateOnly=True or
    level3Evidence. The novelty probe never promotes the ladder."""
    for method in ("trigram", "dag", "auto"):
        out = lean_backend.novelty_check(
            "induction n with | zero => rfl | succ n ih => rw [ih]", corpus=[], method=method
        )
        assert "candidateOnly" not in out
        assert "level3Evidence" not in out


def main() -> int:
    test_dag_commutative_reorder_hashes_identically()
    test_dag_non_commutative_rewrite_keeps_order()
    test_dag_distinct_paths_hash_differently()
    test_dag_corpus_duplicate_detected()
    test_dag_abstains_on_unparseable()
    test_dag_abstains_on_comment_only()
    test_novelty_check_trigram_method_legacy_shape()
    test_novelty_check_default_is_auto_and_uses_dag()
    test_novelty_check_auto_does_not_overclaim_on_unparseable()
    test_novelty_results_carry_no_capability_claim()
    print("test_novelty_dag: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
