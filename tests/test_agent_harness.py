#!/usr/bin/env python3
"""Tests for the agent harness (offline via the mock model provider)."""

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


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


def _isolated_store(monkey_dir: Path) -> None:
    h.RUNS_DIR = monkey_dir  # redirect run logs to a temp dir


def test_happy_path_runs_and_persists() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        task = h.AgentTask(goal="Should we launch on HN this week?", mode="advisor", task_id="t-happy")
        result = h.run_agent(task, client=_mock_client(), max_retries=1)
    assert result.ok is True
    assert result.final_text.strip()
    assert all(s.ok for s in result.steps)
    # decision log persisted with task_start/plan/model_call/task_end events
    events = [json.loads(line) for line in Path(result.trace_path).read_text().splitlines() if line.strip()]
    types = {e["type"] for e in events}
    assert {"task_start", "plan", "model_call", "critic", "task_end"} <= types


def test_failure_classification_and_retry_exhaustion() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)

    def always_fail(text, task, step):
        return {"passed": False, "reasons": ["forced fail"], "detail": {}}

    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        task = h.AgentTask(goal="do the impossible", mode="advisor", task_id="t-fail")
        result = h.run_agent(task, client=_mock_client(), verifier=always_fail, max_retries=2)
    assert result.ok is False
    assert result.steps[0].failure_class == "max_retries_exhausted"
    assert result.steps[0].attempts == 3  # 1 + 2 retries
    assert any("max_retries_exhausted" in f for f in result.failures)


def test_empty_output_is_classified() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = ""  # force empty model output
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h.RUNS_DIR = Path(tmp)
            task = h.AgentTask(goal="produce nothing", mode="advisor", task_id="t-empty")
            result = h.run_agent(task, client=_mock_client(), max_retries=0)
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    assert result.ok is False
    assert result.steps[0].failure_class in {"empty_output", "max_retries_exhausted"}


def test_checkpoint_resume_skips_completed() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        task = h.AgentTask(goal="resumable task", mode="advisor", task_id="t-resume")
        first = h.run_agent(task, client=_mock_client(), max_retries=1)
        assert first.ok is True
        # resume: the completed step is skipped, run still ok
        second = h.run_agent(task, client=_mock_client(), max_retries=1, resume=True)
    assert second.ok is True
    events = [json.loads(line) for line in Path(second.trace_path).read_text().splitlines() if line.strip()]
    assert any(e["type"] == "step_skip" for e in events)


def test_skill_injection_into_task() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    from agent import skills

    skill = skills.get("research-rag")
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        task = h.AgentTask(goal="who wrote the Dao De Jing with sources", mode="advisor", task_id="t-skill", skill=skill)
        result = h.run_agent(task, client=_mock_client(), max_retries=1)
        events = [json.loads(line) for line in Path(result.trace_path).read_text().splitlines() if line.strip()]
    start = next(e for e in events if e["type"] == "task_start")
    assert start["skill"] == "research-rag"
    assert result.ok is True


def test_classify_failure_taxonomy() -> None:
    bad = m.ModelResult(text="", provider="mock", model="m", ok=False, error="boom")
    assert h.classify_failure(result=bad, gate=None, verifier=None, tool_results=None) == "model_error"
    empty = m.ModelResult(text="", provider="mock", model="m", ok=True)
    assert h.classify_failure(result=empty, gate=None, verifier=None, tool_results=None) == "empty_output"
    assert h.classify_failure(result=m.ModelResult(text="x", provider="m", model="m"), gate=None, verifier=None, tool_results=[{"ok": False}]) == "tool_error"
    assert h.classify_failure(result=m.ModelResult(text="x", provider="m", model="m"), gate={"passed": False}, verifier=None, tool_results=None) == "gate_violation"


def main() -> int:
    test_happy_path_runs_and_persists()
    test_failure_classification_and_retry_exhaustion()
    test_empty_output_is_classified()
    test_checkpoint_resume_skips_completed()
    test_skill_injection_into_task()
    test_classify_failure_taxonomy()
    print("test_agent_harness: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
