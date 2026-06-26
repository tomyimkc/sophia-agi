#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for trace distillation -> preference pairs (offline, deterministic)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402
from agent import trace_distill as td  # noqa: E402

_GOOD = "[ok] Analysis.\nDecision: proceed. source discipline noted.\n中文摘要: 完成。"


def _events_fail_then_fix() -> list[dict]:
    return [
        {"type": "task_start", "goal": "compute X", "taskId": "t1"},
        {"type": "step_output", "step": "s1", "attempt": 1, "passed": False, "failureClass": "gate_violation", "output": "wrong answer"},
        {"type": "step_output", "step": "s1", "attempt": 2, "passed": True, "failureClass": None, "output": "right answer"},
    ]


def test_fail_then_fix_yields_one_pair() -> None:
    pairs = td.distill_events(_events_fail_then_fix())
    assert len(pairs) == 1
    p = pairs[0]
    assert p.chosen == "right answer"
    assert p.rejected == "wrong answer"
    assert p.rejected_failure_class == "gate_violation"
    assert p.goal == "compute X" and p.task_id == "t1" and p.step_id == "s1"


def test_only_pass_or_only_fail_yields_nothing() -> None:
    only_pass = [
        {"type": "task_start", "goal": "g"},
        {"type": "step_output", "step": "s1", "attempt": 1, "passed": True, "output": "ok"},
    ]
    only_fail = [
        {"type": "task_start", "goal": "g"},
        {"type": "step_output", "step": "s1", "attempt": 1, "passed": False, "failureClass": "empty_output", "output": "x"},
        {"type": "step_output", "step": "s1", "attempt": 2, "passed": False, "failureClass": "empty_output", "output": "y"},
    ]
    assert td.distill_events(only_pass) == []
    assert td.distill_events(only_fail) == []


def test_empty_outputs_are_skipped() -> None:
    events = [
        {"type": "task_start", "goal": "g"},
        {"type": "step_output", "step": "s1", "attempt": 1, "passed": False, "failureClass": "empty_output", "output": ""},
        {"type": "step_output", "step": "s1", "attempt": 2, "passed": True, "output": "good"},
    ]
    # rejected output is blank -> no usable contrast -> skip.
    assert td.distill_events(events) == []


def test_to_record_and_jsonl_shape() -> None:
    pairs = td.distill_events(_events_fail_then_fix())
    rec = pairs[0].to_record()
    assert rec["prompt"] == "compute X" and rec["chosen"] == "right answer" and rec["rejected"] == "wrong answer"
    assert rec["meta"]["rejectedFailureClass"] == "gate_violation"
    line = td.to_jsonl(pairs)
    assert json.loads(line)["chosen"] == "right answer"


def test_distill_file_skips_corrupt_lines() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "run.jsonl"
        lines = [json.dumps(e) for e in _events_fail_then_fix()]
        lines.insert(1, "{not valid json")  # a corrupt line in the middle
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        pairs = td.distill_file(path)
    assert len(pairs) == 1 and pairs[0].chosen == "right answer"


class _FailThenFixClient:
    """A non-empty-but-failing draft until the harness injects a reflection on
    retry, then a passing answer carrying the 'FIXED' marker. Produces a real
    fail-then-fix trace (with a non-blank rejected attempt) through the live loop."""

    def generate(self, system: str, user: str):
        if "Reflection on the previous failed attempt" in user:
            return m.ModelResult(text="FIXED. " + _GOOD, provider="stub", model="stub", ok=True)
        if "You are a critic" in system:  # the reflect() call — emit a real hint to inject on retry
            return m.ModelResult(text="- include the FIXED marker", provider="stub", model="stub", ok=True)
        return m.ModelResult(text="draft attempt without the marker", provider="stub", model="stub", ok=True)


def _fixed_marker_verifier(text, task, step) -> dict:
    return {"passed": "FIXED" in text, "reasons": [] if "FIXED" in text else ["missing FIXED marker"], "detail": {}}


def test_integration_distills_a_real_harness_trace() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        task = h.AgentTask(goal="solve the thing", mode="advisor", task_id="t-distill")
        result = h.run_agent(task, client=_FailThenFixClient(), verifier=_fixed_marker_verifier, max_retries=2)
        assert result.ok is True  # fixed on retry
        pairs = td.distill_file(result.trace_path)
        # Non-empty failing draft first, passing answer after reflection -> one pair.
        assert len(pairs) >= 1
        p = pairs[0]
        assert "FIXED" in p.chosen and p.rejected == "draft attempt without the marker"
        # a real failure class was captured (gate fires before the custom verifier in the taxonomy)
        assert p.rejected_failure_class in {"verifier_fail", "gate_violation"}
        assert p.goal == "solve the thing"


def main() -> int:
    test_fail_then_fix_yields_one_pair()
    test_only_pass_or_only_fail_yields_nothing()
    test_empty_outputs_are_skipped()
    test_to_record_and_jsonl_shape()
    test_distill_file_skips_corrupt_lines()
    test_integration_distills_a_real_harness_trace()
    print("test_trace_distill: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
