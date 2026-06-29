#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the long-horizon execution eval (eval/long_horizon/).

Covers: deterministic checkpoint logic, horizon-length (longest fully-correct prefix),
completion-rate + CI math, fail-closed dependent steps, and that --emit-pending writes a
not-run / NO-GO-labelled artifact. Uses the committed deterministic mock agents — no model,
no network."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.long_horizon import example_tasks, run_task, run_tasks
from eval.long_horizon.harness import HarnessResult, TaskRun, horizon_length
from eval.long_horizon.tasks import (
    FailAtStepAgent,
    LongHorizonTask,
    PerfectMockAgent,
    Step,
    StepResult,
    expect_int,
    expect_state_sum,
)
from tools import run_long_horizon_eval


# --------------------------------------------------------------------------- #
# Checkpoint logic
# --------------------------------------------------------------------------- #


def test_expect_int_matches_last_integer_and_records():
    state: dict = {}
    cp = expect_int(42, record_as="v")
    assert cp("the answer is 42", state) is True
    assert state["v"] == 42
    # wrong value does not pass and does not record
    state2: dict = {}
    assert cp("the answer is 41", state2) is False
    assert "v" not in state2


def test_expect_state_sum_is_fail_closed_without_prereq():
    cp = expect_state_sum("a", "b", equals_key="c")
    # prerequisite state missing -> must fail closed, even if the number is "right"
    assert cp("0", {}) is False
    # with both prereqs recorded, the dependent step verifies against their sum
    state = {"a": 120, "b": 80}
    assert cp("combined is 200", state) is True
    assert state["c"] == 200
    assert cp("combined is 199", {"a": 120, "b": 80}) is False


def test_checkpoints_are_deterministic():
    cp = expect_int(7)
    out = "result 7"
    assert all(cp(out, {}) for _ in range(5))


# --------------------------------------------------------------------------- #
# Horizon length
# --------------------------------------------------------------------------- #


def test_horizon_length_is_longest_correct_prefix():
    sr = [
        StepResult("a", "", True),
        StepResult("b", "", True),
        StepResult("c", "", False),
        StepResult("d", "", True),  # a later pass must NOT count past the first failure
    ]
    assert horizon_length(sr) == 2


def test_horizon_length_all_pass_equals_length():
    sr = [StepResult(str(i), "", True) for i in range(5)]
    assert horizon_length(sr) == 5


def test_horizon_length_first_fail_is_zero():
    sr = [StepResult("a", "", False), StepResult("b", "", True)]
    assert horizon_length(sr) == 0


# --------------------------------------------------------------------------- #
# Task / harness execution with the deterministic mocks
# --------------------------------------------------------------------------- #


def test_perfect_agent_completes_every_example_task():
    res = run_tasks(PerfectMockAgent(), example_tasks())
    for run in res.runs:
        assert run.completed is True
        assert run.horizon == run.length
    summary = res.summary()
    assert summary["completionRate"]["mean"] == 1.0
    assert summary["stepLevelSuccess"]["mean"] == 1.0


def test_fail_at_step_ends_horizon_and_propagates_to_dependents():
    # The stateful tool chain: fail at 'mul2' (the 3rd step). Horizon = 2 (seed, add3),
    # and the dependent steps after it fail closed because their prereq state is missing.
    agent = FailAtStepAgent(fail_step_id="mul2")
    task = next(t for t in example_tasks() if t.task_id == "synth-stateful-tool-sequence")
    run = run_task(agent, task)
    assert run.horizon == 2
    assert run.completed is False
    by_id = {sr.step_id: sr.passed for sr in run.step_results}
    assert by_id["seed"] is True and by_id["add3"] is True
    assert by_id["mul2"] is False
    # sub4 / add10 depend on the (never recorded) post-mul2 state -> fail closed
    assert by_id["sub4"] is False and by_id["add10"] is False


def test_completion_rate_and_ci_math():
    # Two tasks complete, one fails -> completion rate 2/3.
    tA = LongHorizonTask("t-pass", "two passing steps", (
        Step("s1", "say 1", expect_int(1)),
        Step("s2", "say 2", expect_int(2)),
    ))
    tFail = LongHorizonTask("t-fail", "second step fails", (
        Step("s1", "say 1", expect_int(1)),
        Step("s2", "say 2", expect_int(2)),
    ))

    class Scripted:
        def __init__(self, answers):
            self.answers = answers

        def act(self, task, step, state):
            return self.answers[(task.task_id, step.step_id)]

    answers = {
        ("t-pass", "s1"): "1", ("t-pass", "s2"): "2",
        ("t-fail", "s1"): "1", ("t-fail", "s2"): "999",
    }
    res = run_tasks(Scripted(answers), [tA, tA, tFail], seed=0)
    summary = res.summary()
    assert summary["completionRate"]["mean"] == round(2 / 3, 4)
    lo, hi = summary["completionRate"]["ci95"]
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= summary["completionRate"]["mean"] <= hi <= 1.0
    # step-level success: 3 of 4 distinct... here 6 steps total, 5 pass -> 5/6
    assert summary["stepLevelSuccess"]["mean"] == round(5 / 6, 4)
    # CI is deterministic (seeded bootstrap)
    res2 = run_tasks(Scripted(answers), [tA, tA, tFail], seed=0)
    assert res2.summary()["completionRate"]["ci95"] == summary["completionRate"]["ci95"]


def test_empty_result_summary_is_safe():
    res = HarnessResult(runs=[])
    s = res.summary()
    assert s["nTasks"] == 0
    assert s["completionRate"]["mean"] == 0.0


# --------------------------------------------------------------------------- #
# PENDING artifact (--emit-pending)
# --------------------------------------------------------------------------- #


def test_emit_pending_writes_not_run_nogo_artifact(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "long-horizon-execution.PENDING.public-report.json"
        monkeypatch.setattr(run_long_horizon_eval, "RESULTS_DIR", Path(td))
        monkeypatch.setattr(run_long_horizon_eval, "PENDING_PATH", out)
        rc = run_long_horizon_eval.main(["--emit-pending"])
        assert rc == 0
        artifact = json.loads(out.read_text(encoding="utf-8"))
        assert artifact["status"] == "not_run"
        assert artifact["verdict"] == "NO-GO"
        assert artifact["go"] is False
        assert artifact["canClaimAGI"] is False
        # No fabricated measured numbers.
        assert artifact["results"] is None
        # Suite shape is described but carries no scores.
        assert len(artifact["taskSuite"]) == len(example_tasks())
        for entry in artifact["taskSuite"]:
            assert set(entry) == {"taskId", "length", "description"}


def test_pending_artifact_has_no_measured_numbers():
    artifact = run_long_horizon_eval.build_pending_artifact()
    blob = json.dumps(artifact)
    assert artifact["results"] is None
    assert "completionRate" in artifact["constructs"]
    # The pre-registration is referenced, and AGI is not claimed.
    assert "measurement_spec.json" in artifact["preregistration"]
    assert "not_run" in blob and "NO-GO" in blob


def test_mock_run_summary_is_returned():
    result = run_long_horizon_eval.run_mock("perfect")
    assert result["summary"]["completionRate"]["mean"] == 1.0
    assert result["deterministic"] is True


def test_model_flag_refuses_and_stays_pending():
    # A real-model spec must not silently fabricate a result.
    rc = run_long_horizon_eval.main(["--model", "openrouter:some/model"])
    assert rc == 2


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
