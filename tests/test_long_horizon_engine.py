#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the long-horizon execution engine (offline, deterministic).

Distinct from tests/test_long_horizon.py, which covers tools/run_long_horizon.py
(the single-run interventions logger). This covers agent/long_horizon.py — the
durable task-tree engine with recovery memory.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402
from agent import long_horizon as lh  # noqa: E402

_GOOD = "[ok] Analysis.\nDecision: proceed. source discipline noted.\n中文摘要: 完成。"


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


class _HintGatedClient:
    """Returns a gate-friendly answer ONLY when a recovery hint is in the prompt;
    otherwise returns empty text (which the harness classifies as a failure). Lets
    us prove the recovery loop deterministically without a real model."""

    def generate(self, system: str, user: str):
        if "Recovery hint" in user:
            return m.ModelResult(text=_GOOD, provider="stub", model="stub", ok=True)
        return m.ModelResult(text="", provider="stub", model="stub", ok=True)


def test_recovery_memory_record_recall_and_signature() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mem = lh.RecoveryMemory(path=Path(tmp) / "rec.jsonl")
        assert mem.recall(signature="anything") is None
        sig = lh._signature("Compute the tricky chained value carefully")
        # Same coarse signature for a reworded-but-equivalent task.
        assert lh._signature("carefully compute the chained tricky value") == sig
        mem.record(signature=sig, failure_class="empty_output", hint="show your work")
        mem.record(signature=sig, failure_class="empty_output", hint="latest hint wins")
        assert mem.recall(signature=sig) == "latest hint wins"
        # Empty signature/hint are no-ops, never raise.
        mem.record(signature="", failure_class="x", hint="ignored")
        assert mem.recall(signature="") is None


def test_linear_chain_completes_and_ledger_is_durable() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        ledger = lh.build_ledger(
            "ship a brief",
            [
                {"id": "a", "goal": "gather sources"},
                {"id": "b", "goal": "draft brief", "deps": ["a"]},
            ],
            ledger_id="lh-chain",
            ledgers_dir=Path(tmp),
        )
        result = lh.run_long_horizon(ledger, client=_mock_client(), recovery=lh.RecoveryMemory(path=Path(tmp) / "r.jsonl"))
        assert result.ok is True
        assert result.completed == ["a", "b"]
        # Ledger persisted to disk and reloadable with state intact.
        reloaded = lh.TaskLedger.load("lh-chain", ledgers_dir=Path(tmp))
        assert reloaded is not None
        assert all(n.status == lh.DONE for n in reloaded.nodes)
        assert reloaded.by_id("b").result_text.strip()


def test_resume_skips_completed_nodes() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        ledger = lh.build_ledger("two", [{"id": "a", "goal": "do a"}, {"id": "b", "goal": "do b"}], ledger_id="lh-res", ledgers_dir=Path(tmp))
        lh.run_long_horizon(ledger, client=_mock_client(), recovery=lh.RecoveryMemory(path=Path(tmp) / "r.jsonl"))
        reloaded = lh.TaskLedger.load("lh-res", ledgers_dir=Path(tmp))
        before = [n.attempts for n in reloaded.nodes]
        # Resume: everything already done -> no new attempts, still ok.
        again = lh.run_long_horizon(reloaded, client=_mock_client(), recovery=lh.RecoveryMemory(path=Path(tmp) / "r.jsonl"))
        assert again.ok is True
        assert [n.attempts for n in reloaded.nodes] == before  # no re-execution


def test_dependency_failure_blocks_dependents_fail_closed() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = ""  # force node 'a' to fail (empty output)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h.RUNS_DIR = Path(tmp)
            ledger = lh.build_ledger(
                "blocked chain",
                [
                    {"id": "a", "goal": "failing prerequisite", "max_retries": 0, "max_steps": 1},
                    {"id": "b", "goal": "depends on a", "deps": ["a"], "max_steps": 1},
                ],
                ledger_id="lh-block",
                ledgers_dir=Path(tmp),
            )
            result = lh.run_long_horizon(ledger, client=_mock_client(), recovery=lh.RecoveryMemory(path=Path(tmp) / "r.jsonl"))
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    assert "a" in result.failed
    assert "b" in result.blocked  # dependent never executed on an unmet prerequisite
    assert result.ok is False
    assert ledger.by_id("b").attempts == 0


def test_recovery_hint_flips_a_failing_sibling_to_success() -> None:
    # Two nodes with the SAME signature. The first fails (no hint yet) and records
    # a recovery hint; the second recalls it, so its context carries the hint and
    # the hint-gated client succeeds. Recovery loop closes within one run.
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        mem = lh.RecoveryMemory(path=Path(tmp) / "rec.jsonl")
        ledger = lh.build_ledger(
            "twin tasks",
            [
                {"id": "first", "goal": "compute the tricky chained value", "max_retries": 0, "max_steps": 1},
                {"id": "second", "goal": "compute the tricky chained value", "max_retries": 0, "max_steps": 1},
            ],
            ledger_id="lh-recover",
            ledgers_dir=Path(tmp),
        )
        result = lh.run_long_horizon(ledger, client=_HintGatedClient(), recovery=mem)
        # First fails (no hint), second succeeds (hint recalled into its context).
        assert "first" in result.failed
        assert "second" in result.completed
        # A hint was actually persisted under the shared signature.
        sig = lh._signature("compute the tricky chained value")
        assert mem.recall(signature=sig) is not None


def main() -> int:
    test_recovery_memory_record_recall_and_signature()
    test_linear_chain_completes_and_ledger_is_durable()
    test_resume_skips_completed_nodes()
    test_dependency_failure_blocks_dependents_fail_closed()
    test_recovery_hint_flips_a_failing_sibling_to_success()
    print("test_long_horizon_engine: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
