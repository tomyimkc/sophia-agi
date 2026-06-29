#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for verifier-gated persistent memory.

Deterministic, offline — a stub verifier proves the durable trust boundary without a model,
plus a real-gate fixture proves the wired ``agent.gate`` behavior.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gated_memory import GatedMemory, offline_invariants  # noqa: E402


def _clean(text, question):
    return (True, [])


def _flag(text, question):
    return (False, ["stub: flagged"])


# --- stub-verifier invariants ------------------------------------------------
def test_clean_claim_stored_and_recalled() -> None:
    m = GatedMemory(":memory:", verifier=_clean)
    r = m.remember("Paris is the capital of France.", question="Capital of France?")
    assert r == {"stored": True, "verdict": "accepted"}
    rows = m.recall()
    assert len(rows) == 1 and rows[0]["text"] == "Paris is the capital of France."


def test_flagged_claim_quarantined_not_recalled_reasons_kept() -> None:
    m = GatedMemory(":memory:", verifier=_flag)
    r = m.remember("A dubious claim.", question="q?", source="s")
    assert r["stored"] is False and r["verdict"] == "held"
    assert r["reasons"] == ["stub: flagged"]
    assert m.recall() == []  # never recallable
    held = m.quarantined()
    assert len(held) == 1
    assert held[0]["reasons"] == ["stub: flagged"]  # reasons retained for audit
    assert held[0]["text"] == "A dubious claim."


def test_recall_never_returns_held_row() -> None:
    m = GatedMemory(":memory:", verifier=lambda t, q: ("BAD" not in t, ["bad"] if "BAD" in t else []))
    m.remember("good", question="q")
    m.remember("BAD", question="q")
    recalled = m.recall()
    assert len(recalled) == 1 and recalled[0]["text"] == "good"
    assert all("BAD" not in row["text"] for row in recalled)


def test_recall_like_filter() -> None:
    m = GatedMemory(":memory:", verifier=_clean)
    m.remember("alpha fact", question="q")
    m.remember("beta fact", question="q")
    assert {r["text"] for r in m.recall("alpha")} == {"alpha fact"}
    assert len(m.recall("fact")) == 2


def test_audit_totals_reconcile() -> None:
    m = GatedMemory(":memory:", verifier=lambda t, q: ("no" not in t, ["x"] if "no" in t else []))
    m.remember("yes one", question="q")
    m.remember("yes two", question="q")
    m.remember("no three", question="q")
    assert m.audit() == {"accepted": 2, "held": 1}


def test_cross_session_persistence() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "mem.db")
        a = GatedMemory(path, verifier=_clean)
        a.remember("Durable across sessions.", question="q", source="sess-1")
        a.close()
        # A brand-new instance on the SAME db file (sibling session) sees the accepted row.
        b = GatedMemory(path, verifier=_flag)
        rows = b.recall()
        assert len(rows) == 1
        assert rows[0]["text"] == "Durable across sessions." and rows[0]["source"] == "sess-1"
        b.close()


def test_held_rows_do_not_persist_into_recall_across_sessions() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "mem.db")
        a = GatedMemory(path, verifier=_flag)
        a.remember("held forever should never surface", question="q")
        a.close()
        b = GatedMemory(path, verifier=_clean)
        assert b.recall() == []  # quarantine never becomes readable context
        assert b.audit() == {"accepted": 0, "held": 1}
        b.close()


# --- real-gate fixture -------------------------------------------------------
def test_real_gate_holds_hallucination_accepts_correction() -> None:
    m = GatedMemory(":memory:")  # verifier=None -> real agent.gate
    bad = m.remember(
        "Confucius wrote the Dao De Jing.",
        question="Did Confucius write the Dao De Jing?",
    )
    assert bad["stored"] is False and bad["verdict"] == "held"
    assert bad["reasons"], "the gate must record why it held the hallucination"

    good = m.remember(
        "No, Confucius did not write the Dao De Jing; it is a Daoist text attributed to Laozi. "
        "This is a common Confucian misconception.",
        question="Did Confucius write the Dao De Jing?",
    )
    assert good["stored"] is True and good["verdict"] == "accepted"

    # The corrected answer is recallable; the hallucination is not.
    texts = [r["text"] for r in m.recall()]
    assert any("did not write" in t for t in texts)
    assert all("Confucius wrote the Dao De Jing." not in t for t in texts)
    assert m.audit() == {"accepted": 1, "held": 1}


# --- module invariants -------------------------------------------------------
def test_offline_invariants() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
