#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/gate_feedback.py (offline, deterministic)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import gate_feedback as gf  # noqa: E402


def _miss_result() -> dict:
    """A gate MISS: gate passed clean, but the judge flagged a hallucination."""
    return {
        "case_id": "m1",
        "label": "false",
        "work": "the Book of Daniel",
        "gold_author": "an anonymous author",
        "claimed_author": "the prophet Daniel",
        "raw": {"hallucinated": True},
        "gated": {"hallucinated": True},
        "gated_action": "clean",
    }


def test_detect_miss_produces_candidate() -> None:
    cand = gf.detect_miss(_miss_result())
    assert cand is not None
    (rid, rec), = cand.items()
    assert rec["canonicalTitleEn"] == "the Book of Daniel"
    # honorific stripped to a salient marker
    assert "Daniel" in rec["doNotAttributeTo"]
    # alt-titles reuse dataset logic ("Daniel" derivable from "Book of Daniel")
    assert "Daniel" in rec["altTitlesEn"]


def test_clean_correct_result_returns_none() -> None:
    # gate clean AND judge did NOT flag -> no miss
    ok = _miss_result()
    ok["gated"] = {"hallucinated": False}
    assert gf.detect_miss(ok) is None


def test_gate_caught_it_returns_none() -> None:
    # the gate already acted (not clean) -> not a MISS even if still flagged
    acted = _miss_result()
    acted["gated_action"] = "abstained"
    assert gf.detect_miss(acted) is None


def test_no_claimed_author_returns_none() -> None:
    nc = _miss_result()
    nc["claimed_author"] = ""
    assert gf.detect_miss(nc) is None


def test_append_and_dedupe() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "gate-pending-records.jsonl"
        cand = gf.detect_miss(_miss_result())
        assert cand is not None

        n1 = gf.append_pending(cand, path)
        assert n1 == 1
        assert path.exists()

        # same miss appended again -> still 1 (deduped)
        n2 = gf.append_pending(cand, path)
        assert n2 == 1
        assert path.read_text(encoding="utf-8").strip().count("\n") == 0

        # a different work -> grows to 2
        other = gf.candidate_record("the Epistle to the Hebrews", "Paul the Apostle")
        n3 = gf.append_pending(other, path)
        assert n3 == 2


def main() -> int:
    test_detect_miss_produces_candidate()
    test_clean_correct_result_returns_none()
    test_gate_caught_it_returns_none()
    test_no_claimed_author_returns_none()
    test_append_and_dedupe()
    print("test_gate_feedback: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
