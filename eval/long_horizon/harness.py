# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Long-horizon eval harness — deterministic, offline, per-step verifiable.

Drives an :class:`~eval.long_horizon.tasks.Agent` over a suite of
:class:`~eval.long_horizon.tasks.LongHorizonTask` s, scoring each step with its
deterministic checkpoint (never an LLM judge), and summarising with three honest
long-horizon constructs and CIs from ``tools/eval_stats.py``:

  * **completion rate** — fraction of tasks fully correct (every checkpoint passed);
  * **step-level success** — fraction of all checkpoints passed across all tasks;
  * **horizon length** — per task, the longest fully-correct *prefix* (one slip ends
    it), reported as a mean with a CI and as a fraction of task length.

The harness is fail-closed in the same spirit as the engine in ``agent/long_horizon.py``:
once a step's checkpoint fails, the fully-correct prefix has ended. We still RUN the
remaining steps and record their pass/fail (so step-level success and per-step reporting
are complete), but they no longer count toward the horizon length, and a dependent
verifier will typically fail closed because its prerequisite state was never recorded.

Determinism: the harness adds no randomness of its own. The only stochastic component is
the bootstrap CI, which is seeded (``tools/eval_stats.bootstrap_ci_paired``) so repeated
runs give identical intervals. With a deterministic agent the entire result is bit-stable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from eval.long_horizon.tasks import Agent, LongHorizonTask, StepResult

# eval_stats lives under tools/ (not an importable package); add it to the path.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import eval_stats  # noqa: E402  (path-dependent import, mirrors other eval modules)


def horizon_length(step_results: "list[StepResult]") -> int:
    """Longest fully-correct PREFIX: number of leading steps that all passed. One slip
    ends the horizon, so this is the count of passed steps before the first failure."""
    n = 0
    for sr in step_results:
        if not sr.passed:
            break
        n += 1
    return n


@dataclass
class TaskRun:
    task_id: str
    length: int
    step_results: "list[StepResult]"

    @property
    def horizon(self) -> int:
        return horizon_length(self.step_results)

    @property
    def completed(self) -> bool:
        return self.horizon == self.length and self.length > 0

    @property
    def steps_passed(self) -> int:
        return sum(1 for sr in self.step_results if sr.passed)

    def to_dict(self) -> dict:
        return {
            "taskId": self.task_id,
            "length": self.length,
            "horizon": self.horizon,
            "completed": self.completed,
            "stepsPassed": self.steps_passed,
            "steps": [
                {"stepId": sr.step_id, "passed": sr.passed}
                for sr in self.step_results
            ],
        }


def run_task(agent: Agent, task: LongHorizonTask) -> TaskRun:
    """Run one task end-to-end. State is threaded through every checkpoint so dependent
    steps can read earlier recorded values; the harness never lets the agent write the
    authoritative state directly (only a passing checkpoint records into it)."""
    state: dict = {}
    results: "list[StepResult]" = []
    for step in task.steps:
        # The agent sees a COPY of state (read-only view); the authoritative state is
        # mutated only by the deterministic checkpoint on a pass.
        output = agent.act(task, step, dict(state))
        passed = bool(step.checkpoint(output, state))
        results.append(StepResult(step_id=step.step_id, output=output, passed=passed))
    return TaskRun(task_id=task.task_id, length=task.length, step_results=results)


@dataclass
class HarnessResult:
    runs: "list[TaskRun]" = field(default_factory=list)
    seed: int = 0

    # ------------------------------------------------------------------ #
    # Aggregate constructs
    # ------------------------------------------------------------------ #

    def completion_indicators(self) -> "list[float]":
        return [1.0 if r.completed else 0.0 for r in self.runs]

    def step_indicators(self) -> "list[float]":
        """Flat list of per-step pass(1)/fail(0) across all tasks."""
        return [1.0 if sr.passed else 0.0 for r in self.runs for sr in r.step_results]

    def horizon_lengths(self) -> "list[float]":
        return [float(r.horizon) for r in self.runs]

    def horizon_fractions(self) -> "list[float]":
        return [r.horizon / r.length if r.length else 0.0 for r in self.runs]

    def _mean(self, xs: "list[float]") -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    def _ci(self, xs: "list[float]") -> "list[float]":
        """Fixed-n paired bootstrap CI (seeded -> deterministic). Returns [lo, hi]."""
        if not xs:
            return [None, None]
        return eval_stats.bootstrap_ci_paired(xs, seed=self.seed)

    def summary(self) -> dict:
        comp = self.completion_indicators()
        steps = self.step_indicators()
        hlen = self.horizon_lengths()
        hfrac = self.horizon_fractions()
        n_tasks = len(self.runs)
        n_steps = len(steps)
        return {
            "nTasks": n_tasks,
            "nSteps": n_steps,
            "completionRate": {
                "mean": self._mean(comp),
                "ci95": self._ci(comp),
                "n": n_tasks,
                "mdeAtN": round(eval_stats.mde_at_n(max(1, n_tasks)), 4),
            },
            "stepLevelSuccess": {
                "mean": self._mean(steps),
                "ci95": self._ci(steps),
                "n": n_steps,
                "mdeAtN": round(eval_stats.mde_at_n(max(1, n_steps)), 4),
            },
            "horizonLength": {
                "mean": self._mean(hlen),
                "ci95": self._ci(hlen),
                "meanFractionOfTask": self._mean(hfrac),
                "n": n_tasks,
            },
            "oracle": (
                "deterministic per-step checkpoints (pure functions over agent "
                "output + threaded task state) — never self-judged, never an LLM judge"
            ),
        }

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "perTask": [r.to_dict() for r in self.runs],
            "seed": self.seed,
            "deterministic": True,
        }


def run_tasks(agent: Agent, tasks: "list[LongHorizonTask]", *, seed: int = 0) -> HarnessResult:
    """Run an agent over a task suite and return the scored :class:`HarnessResult`."""
    return HarnessResult(runs=[run_task(agent, t) for t in tasks], seed=seed)
