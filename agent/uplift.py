# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Harness-uplift benchmark — does the harness make a *fixed* model more capable?

The number the DeepSeek-style "Model + Harness = Agent" thesis actually turns on is
not "how good is the model" but "how much does the harness add on top of a fixed
model." This module measures exactly that, holding the model constant:

    uplift = passRate(model + harness) − passRate(model alone)

Both conditions are graded by the SAME external verifier (the harness never grades
itself here), and the per-case results are *paired* (same goal, same model, same
verifier) so the delta isolates the harness loop — plan → execute → critic →
reflect/retry — as the only changing variable.

Honesty discipline (no overclaim — see RESULTS.md):
  * the headline is the **paired** uplift with a **bootstrap 95% CI**; if the CI
    lower bound is <= 0 we report ``demonstrated=False`` — a positive point
    estimate on a tiny suite is NOT a demonstrated effect;
  * deterministic and offline-testable via the mock provider (seeded bootstrap),
    so the measurement itself is reproducible.

This is the eval half of the model<->harness co-evolution loop; ``trace_distill``
is the data half (harness traces -> preference pairs for the next model).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.harness import AgentTask, gate_verifier, run_agent
from agent.model import ModelClient, default_client
from agent.prompts import MODE_PROMPTS

# A verifier matches the harness contract: (text, task, step) -> {"passed", ...}.
Verifier = Callable[[str, AgentTask, dict], dict]


@dataclass
class CaseOutcome:
    case_id: str
    bare_passed: bool
    harness_passed: bool
    delta: int  # harness_passed - bare_passed in {-1, 0, 1}


@dataclass
class UpliftResult:
    suite_size: int
    provider: str
    bare_pass_rate: float
    harness_pass_rate: float
    uplift: float
    uplift_ci95: tuple[float, float]
    demonstrated: bool  # CI lower bound > 0
    cases: list[CaseOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "suiteSize": self.suite_size,
            "provider": self.provider,
            "barePassRate": round(self.bare_pass_rate, 4),
            "harnessPassRate": round(self.harness_pass_rate, 4),
            "uplift": round(self.uplift, 4),
            "upliftCi95": [round(self.uplift_ci95[0], 4), round(self.uplift_ci95[1], 4)],
            "demonstrated": self.demonstrated,
            "cases": [
                {"id": c.case_id, "bare": c.bare_passed, "harness": c.harness_passed, "delta": c.delta}
                for c in self.cases
            ],
        }


def bare_answer(task: AgentTask, *, client: ModelClient) -> str:
    """The model-alone baseline: ONE generation with the mode system prompt and the
    goal — no plan, no tools, no critic, no retry. This is the fair counterfactual
    the harness is measured against (same model, same goal)."""
    system = MODE_PROMPTS.get(task.mode, MODE_PROMPTS["advisor"])
    user = f"## Goal\n{task.goal}\n\nEnd with a Decision section and a short 中文摘要."
    if task.context:
        user = f"## Context\n{task.context}\n\n" + user
    result = client.generate(system, user)
    return result.text if result.ok else ""


def _bootstrap_ci(deltas: list[int], *, seed: int, resamples: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of the paired per-case deltas. Seeded,
    so the interval is reproducible. With n<2 the CI is undefined -> (delta, delta)."""
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(deltas) / n
    if n < 2:
        return (mean, mean)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (lo, hi)


def measure_uplift(
    suite: list[dict[str, Any]],
    *,
    client: ModelClient | None = None,
    provider: str | None = None,
    verifier_for: Callable[[dict], Verifier] | None = None,
    max_retries: int = 2,
    bootstrap_seed: int = 0,
) -> UpliftResult:
    """Run each suite case twice — bare model vs full harness — grade both with the
    SAME external verifier, and return the paired uplift with a bootstrap CI.

    ``suite`` items: ``{"id", "goal", "mode"?, "mustInclude"?}``. ``verifier_for``
    maps a case to its verifier (defaults to the epistemic gate, optionally AND-ed
    with a keyword check when the case carries ``mustInclude``)."""
    client = client or default_client(provider)
    verifier_for = verifier_for or _default_verifier_for
    cases: list[CaseOutcome] = []
    for case in suite:
        verifier = verifier_for(case)
        task = AgentTask(goal=case["goal"], mode=case.get("mode", "advisor"), task_id=f"uplift-{case['id']}")
        step = {"id": "s1", "description": task.goal, "action": "model", "tool": ""}

        bare_text = bare_answer(task, client=client)
        bare_ok = bool(bare_text.strip()) and verifier(bare_text, task, step).get("passed", False)

        outcome = run_agent(task, client=client, verifier=verifier, max_retries=max_retries)
        harness_ok = bool(outcome.ok)

        cases.append(CaseOutcome(case["id"], bare_ok, harness_ok, int(harness_ok) - int(bare_ok)))

    n = len(cases) or 1
    bare_rate = sum(c.bare_passed for c in cases) / n
    harness_rate = sum(c.harness_passed for c in cases) / n
    deltas = [c.delta for c in cases]
    uplift = sum(deltas) / n
    ci = _bootstrap_ci(deltas, seed=bootstrap_seed)
    return UpliftResult(
        suite_size=len(cases),
        provider=provider or "auto",
        bare_pass_rate=bare_rate,
        harness_pass_rate=harness_rate,
        uplift=uplift,
        uplift_ci95=ci,
        demonstrated=ci[0] > 0,
        cases=cases,
    )


def _default_verifier_for(case: dict) -> Verifier:
    must = case.get("mustInclude") or []
    if not must:
        return gate_verifier

    def _verify(text: str, task: AgentTask, step: dict) -> dict:
        gate = gate_verifier(text, task, step)
        lowered = text.lower()
        missing = [kw for kw in must if kw.lower() not in lowered]
        return {
            "passed": gate["passed"] and not missing,
            "reasons": list(gate.get("reasons", [])) + [f"missing: {kw}" for kw in missing],
            "detail": {"missing": missing},
        }

    return _verify
