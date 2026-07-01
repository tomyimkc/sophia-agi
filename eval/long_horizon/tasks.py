# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Long-horizon task model — ordered dependent steps with deterministic checkpoints.

A :class:`LongHorizonTask` is an *ordered* list of :class:`Step` s. Each step carries a
:class:`Checkpoint`: a **pure function** over the agent's output for that step plus the
accumulated task state, returning pass/fail. The whole task is "complete" only when
EVERY checkpoint passes in order — the property that makes a horizon hard is that one
slip fails the task and ends the fully-correct prefix.

Checkpoints are deterministic verifiers, NOT an LLM judge: this is the same discipline
as ``agent/horizon.py`` (independent recomputation / exact-match), extended to a
per-step, stateful chain. The contract:

    checkpoint(output: str, state: dict) -> bool      # pure, no I/O, deterministic

``state`` is a plain dict the harness threads through the task: a checkpoint may read
prior steps' recorded values and (on pass) record its own, so later steps can depend on
earlier results — a verifiable dependency chain rather than independent questions.

The :class:`Agent` interface the harness drives is intentionally tiny (one method) so a
real engine (e.g. ``agent/long_horizon.py`` via a thin adapter) OR a deterministic mock
can satisfy it. Example tasks below are small and clearly SYNTHETIC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

# A checkpoint verifier: pure function (output, state) -> bool. It may mutate ``state``
# to record values later steps depend on, but must do no I/O and be deterministic.
Checkpoint = Callable[[str, dict], bool]


@dataclass(frozen=True)
class Step:
    """One sub-goal in a dependent chain.

    ``step_id``    stable id (for per-step reporting / dependency references).
    ``prompt``     the instruction handed to the agent for this step.
    ``checkpoint`` deterministic verifier over (agent output, task state).
    ``depends_on`` ids of earlier steps whose state this step builds on (documentation
                   + a structural check; ordering is already implied by list position).
    """

    step_id: str
    prompt: str
    checkpoint: Checkpoint
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class LongHorizonTask:
    task_id: str
    description: str
    steps: tuple[Step, ...]

    def __post_init__(self) -> None:
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.task_id}: duplicate step ids {ids}")
        seen: set[str] = set()
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in seen:
                    raise ValueError(
                        f"{self.task_id}: step '{step.step_id}' depends on '{dep}' "
                        "which is not an earlier step (forward/unknown dependency)"
                    )
            seen.add(step.step_id)

    @property
    def length(self) -> int:
        return len(self.steps)


@dataclass
class StepResult:
    step_id: str
    output: str
    passed: bool


class Agent(Protocol):
    """The narrow interface the harness drives over a task.

    ``act`` is called once per step, in order, with the step prompt and a *read-only
    view* of the accumulated task state (the harness owns the authoritative state and
    updates it only via the step's checkpoint). It returns the agent's textual output
    for that step. A real engine adapter or a deterministic mock both satisfy this.
    """

    def act(self, task: "LongHorizonTask", step: "Step", state: dict) -> str:
        """Return the agent's textual output for this step."""


# --------------------------------------------------------------------------- #
# Deterministic checkpoint helpers (pure functions)
# --------------------------------------------------------------------------- #


def _last_int(text: str) -> "int | None":
    nums = re.findall(r"-?\d+", text or "")
    return int(nums[-1]) if nums else None


def expect_int(value: int, *, record_as: str | None = None) -> Checkpoint:
    """Checkpoint: the final integer in the output equals ``value``. Optionally record
    the verified value into ``state[record_as]`` for a later dependent step."""

    def check(output: str, state: dict) -> bool:
        ok = _last_int(output) == value
        if ok and record_as:
            state[record_as] = value
        return ok

    return check


def expect_contains_all(tokens: "tuple[str, ...]", *, record_as: str | None = None) -> Checkpoint:
    """Checkpoint: every token (case-insensitive) appears in the output."""
    lowered = tuple(t.lower() for t in tokens)

    def check(output: str, state: dict) -> bool:
        low = (output or "").lower()
        ok = all(t in low for t in lowered)
        if ok and record_as:
            state[record_as] = list(tokens)
        return ok

    return check


def expect_state_sum(*keys: str, equals_key: str) -> Checkpoint:
    """Checkpoint (dependent): the output's final int equals the sum of prior recorded
    state values under ``keys`` — i.e. this step is only correct if the EARLIER steps
    recorded the right values. Records the result under ``equals_key``."""

    def check(output: str, state: dict) -> bool:
        if any(k not in state for k in keys):
            return False  # a prerequisite step did not pass — fail closed
        target = sum(int(state[k]) for k in keys)
        ok = _last_int(output) == target
        if ok:
            state[equals_key] = target
        return ok

    return check


def expect_cited(claim_token: str, source_token: str) -> Checkpoint:
    """Checkpoint (synthesize-then-cite): the output must mention the synthesised claim
    token AND attribute it to the required source token (a deterministic stand-in for a
    citation-fidelity verifier — NOT an LLM judge)."""

    def check(output: str, state: dict) -> bool:
        low = (output or "").lower()
        return claim_token.lower() in low and source_token.lower() in low

    return check


# --------------------------------------------------------------------------- #
# Synthetic example tasks (small, clearly synthetic)
# --------------------------------------------------------------------------- #


def _retrieve_synthesize_cite_task() -> LongHorizonTask:
    """A 4-step retrieve -> retrieve -> synthesize -> cite chain over a tiny synthetic
    corpus. Step 3 depends on steps 1+2; step 4 depends on step 3."""
    return LongHorizonTask(
        task_id="synth-retrieve-synthesize-cite",
        description=(
            "SYNTHETIC: retrieve two facts from a fixed mini-corpus, synthesize their "
            "sum, then cite the source. Dependent: synthesis needs both retrievals."
        ),
        steps=(
            Step(
                step_id="retrieve_a",
                prompt="From doc A, report the population count (an integer).",
                checkpoint=expect_int(120, record_as="pop_a"),
            ),
            Step(
                step_id="retrieve_b",
                prompt="From doc B, report the population count (an integer).",
                checkpoint=expect_int(80, record_as="pop_b"),
            ),
            Step(
                step_id="synthesize",
                prompt="Report the combined population (sum of A and B).",
                checkpoint=expect_state_sum("pop_a", "pop_b", equals_key="combined"),
                depends_on=("retrieve_a", "retrieve_b"),
            ),
            Step(
                step_id="cite",
                prompt="State the combined population and attribute it to doc A and doc B.",
                checkpoint=expect_cited("200", "doc"),
                depends_on=("synthesize",),
            ),
        ),
    )


def _stateful_tool_sequence_task() -> LongHorizonTask:
    """A 5-step stateful tool sequence: chained arithmetic where each step depends on
    the prior recorded value. One slip fails the rest (fail-closed dependency)."""

    def step_op(prev_key: str, k: int, op: str, record_as: str) -> Checkpoint:
        def check(output: str, state: dict) -> bool:
            if prev_key not in state:
                return False
            base = int(state[prev_key])
            target = base + k if op == "+" else base * k if op == "*" else base - k
            ok = _last_int(output) == target
            if ok:
                state[record_as] = target
            return ok

        return check

    return LongHorizonTask(
        task_id="synth-stateful-tool-sequence",
        description=(
            "SYNTHETIC: a 5-step stateful tool chain (seed -> +3 -> *2 -> -4 -> +10). "
            "Each step depends on the previous recorded value; one slip ends the horizon."
        ),
        steps=(
            Step("seed", "Initialise the accumulator to 5.", expect_int(5, record_as="v0")),
            Step("add3", "Add 3 to the accumulator.", step_op("v0", 3, "+", "v1"), ("seed",)),
            Step("mul2", "Multiply the accumulator by 2.", step_op("v1", 2, "*", "v2"), ("add3",)),
            Step("sub4", "Subtract 4 from the accumulator.", step_op("v2", 4, "-", "v3"), ("mul2",)),
            Step("add10", "Add 10 to the accumulator.", step_op("v3", 10, "+", "v4"), ("sub4",)),
        ),
    )


def _plan_then_check_task() -> LongHorizonTask:
    """A 3-step plan -> enumerate -> conclude chain mixing token and integer checks."""
    return LongHorizonTask(
        task_id="synth-plan-enumerate-conclude",
        description=(
            "SYNTHETIC: state a plan with the required keywords, enumerate exactly 3 "
            "items, then conclude with their count. Conclusion depends on enumeration."
        ),
        steps=(
            Step(
                "plan",
                "State a plan mentioning 'gather', 'verify', and 'report'.",
                expect_contains_all(("gather", "verify", "report")),
            ),
            Step(
                "enumerate",
                "List exactly three sources: alpha, beta, gamma.",
                expect_contains_all(("alpha", "beta", "gamma"), record_as="sources"),
                ("plan",),
            ),
            Step(
                "conclude",
                "State how many sources you listed.",
                _conclude_count("sources", 3),
                ("enumerate",),
            ),
        ),
    )


def _conclude_count(sources_key: str, expected: int) -> Checkpoint:
    def check(output: str, state: dict) -> bool:
        if sources_key not in state:
            return False
        return _last_int(output) == expected

    return check


def example_tasks() -> "list[LongHorizonTask]":
    """The small, synthetic example suite used by the harness self-test and tests."""
    return [
        _retrieve_synthesize_cite_task(),
        _stateful_tool_sequence_task(),
        _plan_then_check_task(),
    ]


# --------------------------------------------------------------------------- #
# Deterministic mock agents (for offline tests + harness self-test)
# --------------------------------------------------------------------------- #


@dataclass
class PerfectMockAgent:
    """A deterministic agent that answers every example step correctly. It recognises
    the synthetic prompts by keyword — a stand-in oracle, NOT a model. Used to prove the
    harness scores a fully-correct trajectory as horizon = task length."""

    _gold: dict = field(default_factory=dict)

    def act(self, task: "LongHorizonTask", step: "Step", state: dict) -> str:
        return _GOLD_ANSWERS[task.task_id][step.step_id](state)


@dataclass
class FailAtStepAgent:
    """Deterministic agent that is perfect until ``fail_step_id`` (per task), where it
    emits a wrong answer. Used to prove horizon-length = longest correct prefix and that
    a dependent later step then fails closed."""

    fail_step_id: str

    def act(self, task: "LongHorizonTask", step: "Step", state: dict) -> str:
        if step.step_id == self.fail_step_id:
            return "WRONG-0xdeadbeef -999999"
        return _GOLD_ANSWERS[task.task_id][step.step_id](state)


# Gold answer generators per example task/step. These read the threaded state so that a
# correct earlier step enables the dependent answer — mirroring the checkpoints exactly.
_GOLD_ANSWERS: "dict[str, dict[str, Callable[[dict], str]]]" = {
    "synth-retrieve-synthesize-cite": {
        "retrieve_a": lambda s: "Doc A population: 120",
        "retrieve_b": lambda s: "Doc B population: 80",
        "synthesize": lambda s: f"Combined population: {int(s.get('pop_a', 0)) + int(s.get('pop_b', 0))}",
        "cite": lambda s: "Combined population is 200 (source: doc A and doc B).",
    },
    "synth-stateful-tool-sequence": {
        "seed": lambda s: "accumulator = 5",
        "add3": lambda s: f"accumulator = {int(s.get('v0', 0)) + 3}",
        "mul2": lambda s: f"accumulator = {int(s.get('v1', 0)) * 2}",
        "sub4": lambda s: f"accumulator = {int(s.get('v2', 0)) - 4}",
        "add10": lambda s: f"accumulator = {int(s.get('v3', 0)) + 10}",
    },
    "synth-plan-enumerate-conclude": {
        "plan": lambda s: "Plan: gather sources, verify each, then report.",
        "enumerate": lambda s: "Sources: alpha, beta, gamma.",
        "conclude": lambda s: f"I listed {len(s.get('sources', []) ) or 3} sources.",
    },
}
