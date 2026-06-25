#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for subagent delegation (offline via the mock model provider)."""

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
from agent import subagent as sa  # noqa: E402


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


def test_fan_out_isolated_traces_and_synthesis() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        specs = [
            sa.SubagentSpec(goal="research the sources", label="research", max_steps=2),
            sa.SubagentSpec(goal="draft the summary", label="draft", max_steps=2),
        ]
        result = sa.delegate("write a sourced brief", specs, client=_mock_client(), parent_id="p1")
        assert result.ok is True
        assert result.n_ok == 2
        assert len(result.children) == 2
        # Each child wrote its OWN isolated trace file.
        trace_paths = {c.trace_path for c in result.children}
        assert len(trace_paths) == 2
        for c in result.children:
            assert Path(c.trace_path).exists()
            assert c.task_id.startswith("p1.sub")
        # Parent delegation trace exists and brackets the children.
        events = [json.loads(l) for l in Path(result.trace_path).read_text().splitlines() if l.strip()]
        types = [e["type"] for e in events]
        assert types[0] == "delegate_start" and types[-1] == "delegate_end"
        assert sum(t == "subagent_done" for t in types) == 2
    assert result.synthesis.strip()


def test_no_successful_children_abstains() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = ""  # force every child to produce empty output
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h.RUNS_DIR = Path(tmp)
            specs = [sa.SubagentSpec(goal="cannot answer", max_steps=1, max_retries=0)]
            result = sa.delegate("impossible parent", specs, client=_mock_client(), parent_id="p2")
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    assert result.ok is False
    assert result.n_ok == 0
    assert result.synthesis == sa.ABSTAIN_NO_CHILDREN  # fail-closed, no invented answer


def test_cost_budget_marks_over_budget_child_not_ok() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        # A zero-dollar ceiling: any real (even mock-zero) run that the harness
        # marks ok is still forced not-ok if it spent anything above the ceiling.
        # Mock cost is 0.0, so use a negative ceiling to force the over-budget path.
        spec = sa.SubagentSpec(goal="bounded child", max_steps=1, cost_budget_usd=-1.0)
        result = sa.delegate("budgeted parent", [spec], client=_mock_client(), parent_id="p3")
    child = result.children[0]
    assert child.over_budget is True
    assert child.ok is False
    assert any("over_budget" in f for f in child.failures)
    assert result.ok is False  # the only child was over budget


def test_tool_scope_is_enforced_fail_closed() -> None:
    # A child scoped to NO tools that nonetheless emits a tool request must fail
    # the step fail-closed (out-of-scope tool refused), not execute the tool.
    os.environ["SOPHIA_MOCK_RESPONSE"] = '```json\n{"tools": ["export_corpus"]}\n```'
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h.RUNS_DIR = Path(tmp)
            spec = sa.SubagentSpec(goal="try a forbidden tool", allowed_tools=set(), max_steps=1, max_retries=0)
            result = sa.delegate("scope test", [spec], client=_mock_client(), parent_id="p4")
            child = result.children[0]
            events = [json.loads(l) for l in Path(child.trace_path).read_text().splitlines() if l.strip()]
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    assert child.ok is False
    # The harness logged a scope block and never ran the tool successfully.
    assert any(e["type"] == "tool_scope_block" for e in events)
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    for e in tool_calls:
        assert all(not r.get("ok") for r in e["results"])


def test_synthesize_false_concatenates_ok_children() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        specs = [
            sa.SubagentSpec(goal="part one", label="one", max_steps=1),
            sa.SubagentSpec(goal="part two", label="two", max_steps=1),
        ]
        result = sa.delegate("two parts", specs, client=_mock_client(), parent_id="p5", synthesize=False)
    assert result.ok is True
    # Plain concatenation of the two successful child outputs (no reduce model call).
    assert result.synthesis == "\n\n".join(c.final_text for c in result.children if c.ok)


def main() -> int:
    test_fan_out_isolated_traces_and_synthesis()
    test_no_successful_children_abstains()
    test_cost_budget_marks_over_budget_child_not_ok()
    test_tool_scope_is_enforced_fail_closed()
    test_synthesize_false_concatenates_ok_children()
    print("test_subagent: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
