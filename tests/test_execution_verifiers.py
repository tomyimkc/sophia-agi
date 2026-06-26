#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the execution/outcome verifier family (offline, deterministic).

Covers:
  * the new agent-domain primitives (tool_result_validity, plan_completion)
  * the FAMILY registry shape + family_report discipline fields
  * verifier_for_task dispatch (code/tool/plan/None fall-through)
  * the closed-loop wedge: a coding case routed to code_tests_pass instead of the gate
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import closed_loop as cl  # noqa: E402
from agent import execution_verifiers as ev  # noqa: E402
from agent import model as m  # noqa: E402


def test_tool_result_validity_passes_when_all_ok() -> None:
    v = ev.tool_result_validity()
    step = {"tool_results": [{"tool": "grep", "ok": True, "output": "x"}]}
    assert v("any text", None, step)["passed"] is True


def test_tool_result_validity_fails_on_tool_error() -> None:
    """A tool error is execution ground truth that the action failed -> fail-closed."""
    v = ev.tool_result_validity()
    step = {"tool_results": [{"tool": "grep", "ok": True}, {"tool": "rm", "ok": False, "error": "denied"}]}
    r = v("text", None, step)
    assert r["passed"] is False
    assert "rm" in r["detail"]["failedTools"]


def test_tool_result_validity_fails_when_no_results() -> None:
    v = ev.tool_result_validity()
    assert v("text", None, {"tool_results": []})["passed"] is False


def test_plan_completion_passes_with_marker() -> None:
    assert ev.plan_completion()("Decision: ship it.", None, {})["passed"] is True


def test_plan_completion_fails_without_marker_or_empty() -> None:
    assert ev.plan_completion()("just some prose", None, {})["passed"] is False
    assert ev.plan_completion()("", None, {})["passed"] is False


def test_family_registry_shape() -> None:
    """The family registry groups execution verifiers by name with factories."""
    assert set(ev.EXECUTION_VERIFIERS) >= {"code_tests_pass", "unit_test", "tool_result_validity", "plan_completion"}
    # each entry is a parameterless factory returning a callable verifier
    for name, factory in ev.EXECUTION_VERIFIERS.items():
        v = factory()
        assert callable(v), f"{name} factory did not return a callable"


def test_family_report_discipline_fields() -> None:
    """No-overclaim: the family report carries candidateOnly/level3Evidence."""
    rep = ev.family_report()
    assert rep["candidateOnly"] is True
    assert rep["level3Evidence"] is False
    assert rep["family"] == "execution_outcome"
    assert rep["count"] == len(ev.EXECUTION_VERIFIERS)


def test_verifier_for_task_dispatch() -> None:
    """code -> code_tests_pass; tool -> all_of(...); plan -> plan_completion; None -> None."""
    from agent.verifiers import all_of

    assert ev.verifier_for_task(ev.TaskSpec(kind="code")) is not None
    tool_v = ev.verifier_for_task(ev.TaskSpec(kind="tool"))
    assert tool_v is not None
    # the tool verifier is a composite (all_of) — check it fails on a tool error
    assert tool_v("Decision: ok", None, {"tool_results": [{"tool": "x", "ok": False}]})["passed"] is False
    assert ev.verifier_for_task(ev.TaskSpec(kind="plan")) is not None
    assert ev.verifier_for_task(ev.TaskSpec(kind=None)) is None  # fall-through -> caller default


def test_closed_loop_routes_coding_case_to_execution_truth() -> None:
    """The wedge: with task_spec_for set, a 'code' case is graded by code_tests_pass,
    not the gate. We force code execution ON and script a client that emits valid
    Python (passes execution) — the loop must score it via execution truth."""

    class _CodeClient:
        """Emits valid python in a fenced block (code_tests_pass extracts + runs it)."""

        def generate(self, system, user):
            text = "Here is the code:\n```python\nassert 2 + 2 == 4\n```"
            return m.ModelResult(text=text, provider="stub", model="stub", ok=True)

    # Save/restore the ORIGINAL value (do not pop): popping a var a sibling test set
    # leaks across the suite — e.g. tests/test_generate_math_code_curriculum.py needs exec
    # ON for its code-row assertions, and an unconditional pop here deleted it, making that
    # test abstain (0 code rows) and spuriously fail in the full-suite ordering.
    _saved = os.environ.get("SOPHIA_ALLOW_CODE_EXEC")
    os.environ["SOPHIA_ALLOW_CODE_EXEC"] = "1"
    try:
        suite = [{"id": "code1", "goal": "Return Python code that asserts 2+2==4.", "mode": "repo"}]

        def spec_for(case):
            return ev.TaskSpec(kind="code")

        with tempfile.TemporaryDirectory() as tmp:
            report = cl.run_closed_loop(
                suite, suite_name="coding",
                make_client=lambda _spec: _CodeClient(),
                initial_spec="base", train_step=cl.noop_train_step,
                runs_root=Path(tmp), max_cycles=1,
                task_spec_for=spec_for,
            )
    finally:
        if _saved is None:
            os.environ.pop("SOPHIA_ALLOW_CODE_EXEC", None)
        else:
            os.environ["SOPHIA_ALLOW_CODE_EXEC"] = _saved
    # The case passed under execution truth (code ran, exit 0) => harness rate 1.0.
    assert report.cycles[0].baseline_harness_rate == 1.0


def test_closed_loop_falls_back_to_gate_when_spec_none() -> None:
    """task_spec_for returning None for a case keeps uplift's default (gate) verifier
    — prior behavior preserved exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        report = cl.run_closed_loop(
            [{"id": "p1", "goal": "Decide the next step.", "mode": "advisor", "mustInclude": ["Decision"]}],
            suite_name="prose",
            make_client=lambda _s: _GateFriendlyClient(),
            initial_spec="base", train_step=cl.noop_train_step,
            runs_root=Path(tmp), max_cycles=1,
            task_spec_for=lambda case: None,  # no execution ground truth -> gate
        )
    # gate-friendly answer passes the gate => harness rate 1.0 (no execution needed)
    assert report.cycles[0].baseline_harness_rate == 1.0


class _GateFriendlyClient:
    _GOOD = "[mock:m] Analysis.\nDecision: proceed (mock). source discipline noted.\n中文摘要: 模拟回答。"

    def generate(self, system, user):
        return m.ModelResult(text=self._GOOD, provider="stub", model="stub", ok=True)


def main() -> int:
    test_tool_result_validity_passes_when_all_ok()
    test_tool_result_validity_fails_on_tool_error()
    test_tool_result_validity_fails_when_no_results()
    test_plan_completion_passes_with_marker()
    test_plan_completion_fails_without_marker_or_empty()
    test_family_registry_shape()
    test_family_report_discipline_fields()
    test_verifier_for_task_dispatch()
    test_closed_loop_routes_coding_case_to_execution_truth()
    test_closed_loop_falls_back_to_gate_when_spec_none()
    print("test_execution_verifiers: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
