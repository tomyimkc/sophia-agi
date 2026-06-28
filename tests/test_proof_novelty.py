# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the semantic proof-novelty assessor (agent.proof_novelty)
and the held-out-fresh theorem split — the explicit guard for THEORY-ISSUES #4.

No network, no model, no Lean. The injected ``judge_fn`` is a FAKE in every test, so
``agent.model.complete`` (the lazy import inside ``make_judge_fn``) is never touched.

The contract these tests lock:
  * surface_novelty ~1.0 for identical proofs, low for distinct proofs.
  * structural_signal flags an induction proof (uses_induction) vs an rfl one
    (only_closers), and flags a cited named lemma.
  * semantic_novelty with a fake judge that says "standard lemma" -> likely_recall True;
    a fake judge that says "non-standard / constructed" -> likely_recall False.
  * An unusable judge (raises / empty) -> likely_recall True (fail-closed).
  * held-out-fresh.jsonl loads, has >= 8 entries, and every entry is core_only with a
    Lean theorem signature ending in ' := by'.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.proof_novelty import (  # noqa: E402
    SURFACE_DUP_THRESHOLD,
    semantic_novelty,
    structural_signal,
    surface_novelty,
)

HELD_OUT_FRESH = ROOT / "formal_proofs" / "eval" / "held-out-fresh.jsonl"


# --------------------------------------------------------------------------- #
# Fake judges (deterministic; no network). Each returns the two-line format the
# parser reads: STANDARD_LEMMA: yes/no and PROOF_RECALL: yes/no.
# --------------------------------------------------------------------------- #


def _judge_standard(_prompt: str) -> str:
    return "STANDARD_LEMMA: yes\nPROOF_RECALL: yes"


def _judge_nonstandard(_prompt: str) -> str:
    return "STANDARD_LEMMA: no\nPROOF_RECALL: no"


def _judge_raises(_prompt: str) -> str:
    raise RuntimeError("judge backend exploded")


def _judge_empty(_prompt: str) -> str:
    return ""


# --------------------------------------------------------------------------- #
# surface_novelty
# --------------------------------------------------------------------------- #


def test_surface_novelty_identical_is_high():
    proof = "induction n with | zero => rfl | succ k ih => rw [Nat.add_succ, ih]"
    overlap = surface_novelty(proof, [proof])
    assert overlap >= 0.99, overlap  # identical -> Jaccard ~1.0
    assert overlap >= SURFACE_DUP_THRESHOLD


def test_surface_novelty_distinct_is_low():
    proof = "induction n with | zero => rfl | succ k ih => rw [Nat.add_succ, ih]"
    corpus = ["exact ⟨hb, ha⟩", "fun p => p"]
    overlap = surface_novelty(proof, corpus)
    assert overlap < SURFACE_DUP_THRESHOLD, overlap


def test_surface_novelty_empty_corpus_is_zero():
    assert surface_novelty("rfl", []) == 0.0


# --------------------------------------------------------------------------- #
# structural_signal
# --------------------------------------------------------------------------- #


def test_structural_signal_flags_induction():
    sig = structural_signal(
        "induction n with | zero => rfl | succ k ih => rw [Nat.add_succ, ih]"
    )
    assert sig["uses_induction"] is True
    assert sig["only_closers"] is False
    # An induction proof should score as more "work done" than a pure closer.
    assert sig["work_score"] > 0.5, sig


def test_structural_signal_flags_rfl_only_closer():
    sig = structural_signal("rfl")
    assert sig["uses_induction"] is False
    assert sig["only_closers"] is True
    assert sig["work_score"] < 0.5, sig


def test_structural_signal_detects_named_lemma():
    sig = structural_signal("rw [Nat.add_comm, List.map_map]")
    assert sig["cites_named_lemma"] is True
    assert "Nat.add_comm" in sig["named_lemmas"]
    assert "List.map_map" in sig["named_lemmas"]


# --------------------------------------------------------------------------- #
# semantic_novelty — the issue-4 verdict
# --------------------------------------------------------------------------- #


def test_semantic_novelty_standard_lemma_is_likely_recall():
    out = semantic_novelty(
        "theorem map_map : (l.map f).map g = l.map (g ∘ f) := by",
        "simp [List.map_map]",
        _judge_standard,
        corpus=[],
    )
    assert out["likely_recall"] is True, out
    assert out["judge_ok"] is True
    assert out["theorem_is_standard"] is True


def test_semantic_novelty_nonstandard_is_not_likely_recall():
    # A bespoke theorem with a constructed multi-step induction proof; fake judge says
    # non-standard + not recall. Distinct from any corpus entry so surface_novel holds.
    out = semantic_novelty(
        "theorem hf01 (n : Nat) : (n + 2) * (n + 1) = n * n + 3 * n + 2 := by",
        "induction n with | zero => decide | succ k ih => rw [Nat.succ_mul]; omega",
        _judge_nonstandard,
        corpus=["fun p => p", "exact ⟨hb, ha⟩"],
    )
    assert out["likely_recall"] is False, out
    assert out["surface_novel"] is True, out
    assert out["judge_ok"] is True
    assert out["theorem_is_standard"] is False
    assert out["proof_is_recall"] is False


def test_semantic_novelty_surface_duplicate_flagged():
    proof = "induction n with | zero => decide | succ k ih => omega"
    out = semantic_novelty("theorem t : True := by", proof, _judge_nonstandard,
                           corpus=[proof])
    assert out["surface_novel"] is False, out  # exact corpus duplicate
    assert out["best_overlap"] >= SURFACE_DUP_THRESHOLD


def test_semantic_novelty_judge_raises_fails_closed():
    out = semantic_novelty("theorem t : True := by", "decide", _judge_raises, corpus=[])
    assert out["judge_ok"] is False
    assert out["likely_recall"] is True, out  # fail-closed: assume recall


def test_semantic_novelty_judge_empty_fails_closed():
    out = semantic_novelty("theorem t : True := by", "decide", _judge_empty, corpus=[])
    assert out["judge_ok"] is False
    assert out["likely_recall"] is True, out


# --------------------------------------------------------------------------- #
# held-out-fresh.jsonl
# --------------------------------------------------------------------------- #


def test_held_out_fresh_loads_and_is_well_formed():
    assert HELD_OUT_FRESH.exists(), HELD_OUT_FRESH
    rows = []
    for line in HELD_OUT_FRESH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    assert len(rows) >= 8, f"expected >= 8 held-out-fresh theorems, got {len(rows)}"
    ids = set()
    for row in rows:
        assert row["core_only"] is True, row
        assert "id" in row and row["id"], row
        ids.add(row["id"])
        assert "rationale" in row and row["rationale"].strip(), row
        stmt = row["statement"]
        assert stmt.startswith("theorem "), stmt
        assert stmt.rstrip().endswith(":= by"), stmt  # Lean 4 signature ending in ' := by'
        assert "import" not in stmt, f"core_only theorem must not import Mathlib: {stmt}"
    assert len(ids) == len(rows), "held-out-fresh ids must be unique"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")
