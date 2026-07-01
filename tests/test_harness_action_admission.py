#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Harness Build-3 action admission: opt-in gating of tool calls. Offline via the mock provider.

Asserts: with an admitter that marks a requested tool high-risk, the harness WITHHOLDS that
tool (it never reaches run_tools) and records a failed result; with no admitter the path is
unchanged (the tool reaches run_tools); and an ordinary tool under an admitter still runs.
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
from agent.gui_action_gate import tool_action_admitter  # noqa: E402

_TOOL = "export_corpus"  # a real TOOL_CATALOG entry


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


def _step() -> dict:
    return {"id": "s1", "description": "do a thing", "action": "tool", "tool": _TOOL}


def _run_step(monkeypatch_run_tools, admit):
    """Drive _execute_step once with a mock client that requests _TOOL, recording run_tools calls."""
    calls: list[list[str]] = []

    def fake_run_tools(tools, *, approved):
        calls.append(list(tools))
        return [{"tool": t, "ok": True, "output": "ok"} for t in tools]

    orig = h.run_tools
    h.run_tools = fake_run_tools
    os.environ["SOPHIA_MOCK_RESPONSE"] = '{"tools": ["%s"]}' % _TOOL
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h.RUNS_DIR = Path(tmp)
            task = h.AgentTask(goal="g", mode="advisor", task_id="t-admit")
            store = h.RunStore(task.task_id, runs_dir=Path(tmp)).fresh()
            result = h._execute_step(
                task, _step(), client=_mock_client(), store=store, verifier=h.gate_verifier,
                prior="", max_retries=0, approve_tools=True, admit_action=admit,
            )
        return result, calls
    finally:
        h.run_tools = orig
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)


def test_high_risk_tool_is_withheld_before_run_tools() -> None:
    admit = tool_action_admitter({_TOOL})  # mark the requested tool high-risk
    result, calls = _run_step(None, admit)
    # the tool was never executed...
    assert calls == [] or _TOOL not in (calls[0] if calls else []), (calls, result.tool_results)
    # ...and a failed result was recorded for it (so the step does not pass)
    assert any(tr["tool"] == _TOOL and not tr["ok"] and "admission" in tr["error"]
               for tr in result.tool_results), result.tool_results


def test_ordinary_tool_runs_under_admitter() -> None:
    admit = tool_action_admitter({"some_other_high_risk_tool"})  # _TOOL is NOT high-risk
    result, calls = _run_step(None, admit)
    assert calls and _TOOL in calls[0], (calls, result.tool_results)


def test_no_admitter_is_unchanged_behaviour() -> None:
    result, calls = _run_step(None, None)  # admit_action=None -> opt-out, original path
    assert calls and _TOOL in calls[0], (calls, result.tool_results)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
