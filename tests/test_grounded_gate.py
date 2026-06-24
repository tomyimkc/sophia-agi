#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/grounded_gate.py (offline, deterministic).

Verifies that on-demand record synthesis (a) resolves a real snapshot work to
its documented author, (b) turns a WRONG attribution of a work absent from the
base corpus into a record that makes ``provenance_faithful`` fire, and (c) does
NOT synthesize anything for a correct attribution (no false positive).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import grounded_gate as gg  # noqa: E402
from agent import verifiers as v  # noqa: E402


def test_resolve_true_author_from_snapshot() -> None:
    # A known single-author work in the real committed snapshot.
    assert gg.resolve_true_author("Candide") == "Voltaire"
    assert gg.resolve_true_author("The Republic") == "Plato"
    # Alt-title form resolves too ("the Aeneid" -> "Aeneid").
    assert gg.resolve_true_author("the Aeneid") == "Virgil"
    # An unknown work resolves to nothing.
    assert gg.resolve_true_author("A Totally Fabricated Nonexistent Treatise") is None


def test_belief_fn_fallback() -> None:
    # Snapshot misses; the passed belief_fn supplies the author.
    def belief(work: str):
        return "Real Author" if "Obscure" in work else None

    assert gg.resolve_true_author("An Obscure Pamphlet", belief_fn=belief) == "Real Author"
    # No belief_fn -> still None (default skips it).
    assert gg.resolve_true_author("An Obscure Pamphlet") is None


def test_synth_fires_on_wrong_attribution() -> None:
    # "Candide" is in the snapshot (gold Voltaire) but NOT in the empty base
    # corpus. A wrong attribution must mint a record that makes the gate fire.
    base: dict = {}
    text = "Rousseau wrote Candide in his later years."
    additions = gg.synth_records_for_claim(text, base_records=base)
    assert additions, "expected a synthesized record for a wrong attribution"

    # The synthesized record has the same shape as build_gate_records.
    (rid, rec), = additions.items()
    assert rec["canonicalTitleEn"] == "Candide"
    assert rec["doNotAttributeTo"] == ["Rousseau"]
    assert "altTitlesEn" in rec

    # Merged into records, provenance_faithful now CATCHES the merge.
    merged = {**base, **additions}
    ver = v.provenance_faithful(merged)
    assert ver(text, None, {})["passed"] is False, "gate should fire on synthesized record"
    # And the correct attribution still passes under that same record.
    assert ver("Voltaire wrote Candide.", None, {})["passed"] is True


def test_no_false_positive_on_correct_attribution() -> None:
    # A CORRECT attribution of a snapshot work yields no synthesized record.
    base: dict = {}
    assert gg.synth_records_for_claim("Voltaire wrote Candide.", base_records=base) == {}
    # Surname/full-name overlap is treated as the same person (no false positive).
    assert gg.synth_records_for_claim("Hobbes wrote Leviathan.", base_records=base) == {}


def test_no_synth_for_unknown_or_covered_work() -> None:
    base: dict = {}
    # Unknown work -> cannot resolve -> nothing synthesized.
    assert gg.synth_records_for_claim(
        "Nobody wrote The Imaginary Codex of Zorbon.", base_records=base
    ) == {}
    # Already covered by the base corpus -> no duplicate synthesis even if wrong.
    covered = {
        "candide": {
            "canonicalTitleEn": "Candide",
            "altTitlesEn": [],
            "doNotAttributeTo": ["Diderot"],
        }
    }
    assert gg.synth_records_for_claim(
        "Rousseau wrote Candide.", base_records=covered
    ) == {}


def test_corrections_and_hedges_do_not_synthesize() -> None:
    base: dict = {}
    # A negation/correction is not a bare assertion -> no synthesis.
    assert gg.synth_records_for_claim(
        "Rousseau did not write Candide.", base_records=base
    ) == {}
    # Ambiguous/multi-author gold is never used as a confident single author.
    # ("Analects" gold is "Confucius (compiled by his disciples)".)
    assert gg.synth_records_for_claim(
        "Mencius wrote the Analects.", base_records=base
    ) == {}


def main() -> int:
    test_resolve_true_author_from_snapshot()
    test_belief_fn_fallback()
    test_synth_fires_on_wrong_attribution()
    test_no_false_positive_on_correct_attribution()
    test_no_synth_for_unknown_or_covered_work()
    test_corrections_and_hedges_do_not_synthesize()
    print("test_grounded_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
