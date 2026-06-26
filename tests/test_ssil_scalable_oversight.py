#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the GSO scalable-oversight gate. Offline, pure stdlib, no torch.

Covers: the promote path via demo_bundle(); each distinct reject reason (panel
tie, winner inconsistent with anchor, margin below minMargin); each distinct
quarantine/abstain reason (no verifiable anchor, weak-to-strong gated, anchor not
a dict, fewer than two answers); fail-closed on missing required inputs; and the
honesty invariants (canClaimAGI False, candidateOnly True, verdict in the legal set).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_scalable_oversight import (  # noqa: E402
    GATE_ID,
    SCHEMA,
    demo_bundle,
    evaluate,
    weak_to_strong,
)

_VERDICTS = {"promote", "quarantine", "reject"}

_ANCHOR = {
    "schema": "sophia.provenance_record.v1",
    "work": "Project Phoenix Charter",
    "author": "the founding committee",
    "year": "2019",
    "independentSources": 3,
}


def _assert_envelope(d: dict) -> None:
    assert d["schema"] == SCHEMA, d
    assert d["gate"] == GATE_ID, d
    assert d["canClaimAGI"] is False, d
    assert d["candidateOnly"] is True, d
    assert d["level3Evidence"] is False, d
    assert d["verdict"] in _VERDICTS, d
    assert isinstance(d["reasons"], list) and d["reasons"], d
    assert isinstance(d["boundary"], str) and d["boundary"], d
    assert "timestamp" in d, d


# --- promote --------------------------------------------------------------- #


def test_promote_via_demo_bundle() -> None:
    d = evaluate(demo_bundle())
    _assert_envelope(d)
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["winnerConsistent"] is True


# --- reject reasons -------------------------------------------------------- #


def test_reject_panel_tie() -> None:
    # Two answers that each cite the same anchor fields equally -> per-judge ties ->
    # no votes cast -> panel tie (margin 0).
    same = "The Project Phoenix Charter, by the founding committee, 2019."
    d = evaluate({"question": "q", "answers": [same, same], "anchor": _ANCHOR})
    _assert_envelope(d)
    assert d["verdict"] == "reject", d
    assert any("panel tie" in r for r in d["reasons"]), d["reasons"]


def test_reject_winner_inconsistent_with_anchor() -> None:
    # The winner echoes only the work title (thin partial overlap) and wins the vote
    # against an answer that overlaps nothing, but it contradicts the anchor on author
    # and year -> coverage below half -> winner inconsistent with the anchor.
    d = evaluate(
        {
            "question": "q",
            "answers": [
                "The Project Phoenix Charter was written by Bob in 1850.",
                "Nobody knows who wrote that document.",
            ],
            "anchor": _ANCHOR,
        }
    )
    _assert_envelope(d)
    assert d["verdict"] == "reject", d
    assert any("inconsistent with anchor" in r for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["winnerConsistent"] is False, d["metrics"]


def test_reject_margin_below_min_margin() -> None:
    # A consistent winner but a high minMargin the panel cannot clear.
    d = evaluate(
        {
            "question": "q",
            "answers": [
                "The Project Phoenix Charter was written by the founding committee in 2019.",
                "The Project Phoenix Charter was written by Alice in 2024.",
            ],
            "anchor": _ANCHOR,
            "judgeCount": 3,
            "minMargin": 99,
        }
    )
    _assert_envelope(d)
    assert d["verdict"] == "reject", d
    assert any("below minMargin" in r for r in d["reasons"]), d["reasons"]


# --- quarantine / abstain reasons ------------------------------------------ #


def test_quarantine_no_verifiable_anchor() -> None:
    d = evaluate({"question": "q", "answers": ["A claim.", "B claim."], "anchor": None})
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any("no verifiable anchor" in r for r in d["reasons"]), d["reasons"]
    assert any(r.startswith("abstained:") for r in d["reasons"]), d["reasons"]


def test_quarantine_weak_to_strong_gated() -> None:
    d = evaluate(
        {
            "question": "q",
            "answers": [
                "The Project Phoenix Charter was written by the founding committee in 2019.",
                "The Project Phoenix Charter was written by Alice in 2024.",
            ],
            "anchor": _ANCHOR,
            "weakToStrong": {"weakConfidence": 0.1, "strongClaim": "superhuman result"},
        }
    )
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any("below floor" in r for r in d["reasons"]), d["reasons"]
    assert any(r.startswith("abstained:") for r in d["reasons"]), d["reasons"]


def test_quarantine_anchor_not_a_dict() -> None:
    d = evaluate({"question": "q", "answers": ["A", "B"], "anchor": "a string anchor"})
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any("anchor must be a fact-record dict" in r for r in d["reasons"]), d["reasons"]


def test_quarantine_fewer_than_two_answers() -> None:
    d = evaluate({"question": "q", "answers": ["only one"], "anchor": _ANCHOR})
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any("at least two candidates" in r for r in d["reasons"]), d["reasons"]


# --- fail-closed: missing required inputs ---------------------------------- #


def test_fail_closed_missing_question() -> None:
    d = evaluate({"answers": ["A", "B"], "anchor": _ANCHOR})
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any("missing required input: question" in r for r in d["reasons"]), d["reasons"]


def test_fail_closed_missing_answers() -> None:
    d = evaluate({"question": "q", "anchor": _ANCHOR})
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any("missing required input: answers" in r for r in d["reasons"]), d["reasons"]


def test_fail_closed_bundle_none() -> None:
    d = evaluate(None)
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any("bundle is None" in r for r in d["reasons"]), d["reasons"]


# --- weak_to_strong helper ------------------------------------------------- #


def test_weak_to_strong_gates_below_floor() -> None:
    gated = weak_to_strong(0.2, "strong claim")
    assert gated["gated"] is True, gated
    clear = weak_to_strong(0.9, "strong claim")
    assert clear["gated"] is False, clear


def test_weak_to_strong_none_abstains_not_crash() -> None:
    # A missing weak signal must NOT crash (float(None) TypeError) and must NOT read as
    # high confidence: it abstains (gated=True) naming the absent signal.
    rec = weak_to_strong(None, "superhuman result")
    assert rec["gated"] is True, rec
    assert rec["weakConfidence"] is None, rec
    assert rec["reason"] == "abstained: no weak-supervisor signal: cannot endorse strong claim", rec
    # Non-numeric garbage resolves the same way rather than raising.
    rec2 = weak_to_strong("not-a-number", "x")
    assert rec2["gated"] is True, rec2


def test_quarantine_weak_to_strong_none_signal() -> None:
    # evaluate's weakToStrong branch with weakConfidence=None must quarantine, never
    # crash, even though an anchor is present.
    d = evaluate(
        {
            "question": "q",
            "answers": [
                "The Project Phoenix Charter was written by the founding committee in 2019.",
                "The Project Phoenix Charter was written by Alice in 2024.",
            ],
            "anchor": _ANCHOR,
            "weakToStrong": {"weakConfidence": None, "strongClaim": "superhuman result"},
        }
    )
    _assert_envelope(d)
    assert d["verdict"] == "quarantine", d
    assert any(
        r == "abstained: no weak-supervisor signal: cannot endorse strong claim"
        for r in d["reasons"]
    ), d["reasons"]
    assert d["metrics"]["weakToStrong"]["gated"] is True, d["metrics"]


# --- honesty invariants over every path ------------------------------------ #


def test_honesty_invariants_hold_on_all_paths() -> None:
    bundles = [
        demo_bundle(),
        {"question": "q", "answers": ["A", "B"], "anchor": None},
        {"question": "q", "answers": ["x", "y"], "anchor": "not a dict"},
        {"answers": ["A", "B"], "anchor": _ANCHOR},
        None,
    ]
    for b in bundles:
        d = evaluate(b)
        assert d["canClaimAGI"] is False, d
        assert d["candidateOnly"] is True, d
        assert d["verdict"] in _VERDICTS, d


def main() -> int:
    test_promote_via_demo_bundle()
    test_reject_panel_tie()
    test_reject_winner_inconsistent_with_anchor()
    test_reject_margin_below_min_margin()
    test_quarantine_no_verifiable_anchor()
    test_quarantine_weak_to_strong_gated()
    test_quarantine_anchor_not_a_dict()
    test_quarantine_fewer_than_two_answers()
    test_fail_closed_missing_question()
    test_fail_closed_missing_answers()
    test_fail_closed_bundle_none()
    test_weak_to_strong_gates_below_floor()
    test_weak_to_strong_none_abstains_not_crash()
    test_quarantine_weak_to_strong_none_signal()
    test_honesty_invariants_hold_on_all_paths()
    print("test_ssil_scalable_oversight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
