#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the gated experience bank (cross-run procedural memory) and its
opt-in read/write seam in the long-horizon engine.

Deterministic, offline — the bank's own invariants run stub-free (pure stdlib);
the long-horizon integration uses the mock model client.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import long_horizon as lh  # noqa: E402
from agent import model as m  # noqa: E402
from agent.experience_memory import (  # noqa: E402
    ExperienceBank,
    format_hints,
    offline_invariants,
    record_from_subagent,
    validate_record,
)


def _bank(tmp: str) -> ExperienceBank:
    return ExperienceBank(path=Path(tmp) / "bank.jsonl",
                          quarantine_path=Path(tmp) / "quarantine.jsonl")


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail


def test_add_requires_verification_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = _bank(tmp)
        # missing verifiedBy → held
        r = bank.add({"task": "t", "outcomeSummary": "s", "verdict": "accepted"})
        assert not r["stored"] and r["verdict"] == "held"
        # non-accepted verdict → held even with evidence
        r2 = bank.add({"task": "t", "outcomeSummary": "s", "verdict": "failed",
                       "verifiedBy": ["gate"]})
        assert not r2["stored"]
        # both quarantined, neither searchable
        assert bank.audit() == {"accepted": 0, "held": 2}
        assert bank.search("t", min_score=0.0) == []


def test_validate_record_reasons_are_specific() -> None:
    reasons = validate_record({"task": "", "verdict": "accepted"})
    joined = " ".join(reasons)
    assert "task" in joined and "verifiedBy" in joined and "outcomeSummary" in joined


def test_search_ranking_and_floor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = _bank(tmp)
        bank.add(record_from_subagent("Check attribution of the Dao De Jing",
                                      "Laozi attribution is traditional, contested.",
                                      verified_by=["gate"], source="t"))
        bank.add(record_from_subagent("Compute the integral of x squared",
                                      "x^3/3 + C, machine-verified.",
                                      verified_by=["math_verifier"], source="t"))
        hits = bank.search("attribution of the Dao De Jing")
        assert hits and "Dao De Jing" in hits[0]["task"]
        # the math record must not outrank the on-topic one
        assert all("integral" not in h["task"] for h in hits[:1])


def test_hints_block_is_advisory_and_bounded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = _bank(tmp)
        for i in range(6):
            bank.add(record_from_subagent(f"solve linear equation variant {i}",
                                          "x = 2, verified." + str(i),
                                          verified_by=["math_verifier"], source="t"))
        hint = bank.hints_for("solve linear equation variant 3")
        assert hint is not None and "advisory only" in hint
        # bounded to at most 3 hints (header + 3 bullets)
        assert len(hint.splitlines()) <= 4


def test_format_hints_names_verifiers() -> None:
    text = format_hints([{"score": 0.9, "task": "t", "outcomeSummary": "s",
                          "verifiedBy": ["gate", "math_verifier"]}])
    assert "gate,math_verifier" in text


# --------------------------------------------------------------------------- #
# long_horizon integration (opt-in seam)
# --------------------------------------------------------------------------- #


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


def test_long_horizon_without_bank_unchanged() -> None:
    """experience=None (default) keeps the engine's behavior identical."""
    with tempfile.TemporaryDirectory() as tmp:
        ledger = lh.build_ledger("g", [{"id": "a", "goal": "say hello"}],
                                 ledger_id="exp-off", ledgers_dir=Path(tmp))
        result = lh.run_long_horizon(ledger, client=_mock_client(),
                                     recovery=lh.RecoveryMemory(path=Path(tmp) / "r.jsonl"))
        assert result.ok


def test_long_horizon_records_and_recalls_experience() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bank = _bank(tmp)
        ledger = lh.build_ledger(
            "g", [{"id": "a", "goal": "summarize the doctrine of the mean"}],
            ledger_id="exp-on", ledgers_dir=Path(tmp))
        result = lh.run_long_horizon(ledger, client=_mock_client(),
                                     recovery=lh.RecoveryMemory(path=Path(tmp) / "r.jsonl"),
                                     experience=bank)
        assert result.ok
        # the verified success was written back with harness evidence
        assert bank.audit()["accepted"] == 1
        rows = bank.search("doctrine of the mean")
        assert rows and rows[0]["verifiedBy"] == ["subagent.run_subagent:ok"]
        assert rows[0]["source"] == "long_horizon:exp-on"
        # and a second, similar run sees it as an advisory hint in node context
        node = ledger.nodes[0]
        ctx = lh.node_context(ledger, node, None,
                              experience_hint=bank.hints_for(node.goal))
        assert "advisory only" in ctx
