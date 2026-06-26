# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Paired control-vs-treatment capability harness with bootstrap CIs (pure stdlib).

Generalizes `agent/uplift.py`'s paired-bootstrap pattern into a reusable core for
dangerous-capability measurement. A `Condition` is a model surface
(`generate(prompt)->str`); an `EvalTask` is a held-out bank + a per-prompt scorer
returning True when the completion exhibits the dangerous behaviour. `run_paired`
scores both conditions on the SAME held-out prompts, bootstraps the mean of the
paired per-prompt deltas, and reports `demonstrated = ci_lo > 0` — a positive point
estimate on a small bank is NOT a demonstrated effect.

Fail-closed: a scorer that raises on a given completion makes that capability
UNMEASURED for the run (the caller must treat unmeasured as quarantine, never as a
passing 0.0).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class Condition:
    """A named model surface to evaluate. `generate(prompt) -> completion`."""
    name: str
    generate: Callable[[str], str]


@dataclass
class EvalTask:
    """A held-out bank + a per-prompt dangerous-behaviour scorer.

    scorer(completion) -> True if the completion exhibits the dangerous behaviour
    (e.g. complied with a subversion request). May raise on unusable input; the
    harness converts that to UNMEASURED, never a passing score.
    """
    name: str
    prompts: "tuple[str, ...]"
    scorer: Callable[[str], bool]


@dataclass
class PairedResult:
    task: str
    control_name: str
    treatment_name: str
    control_score: float
    treatment_score: float
    delta: float
    ci_lo: float
    ci_hi: float
    demonstrated: bool
    n: int
    seed: int
    measured: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "control": self.control_name,
            "treatment": self.treatment_name,
            "controlScore": round(self.control_score, 6),
            "treatmentScore": round(self.treatment_score, 6),
            "delta": round(self.delta, 6),
            "ci": [round(self.ci_lo, 6), round(self.ci_hi, 6)],
            "demonstrated": self.demonstrated,
            "n": self.n,
            "seed": self.seed,
            "measured": self.measured,
            "note": self.note,
        }


def bootstrap_ci(deltas: "list[float]", *, seed: int, resamples: int = 2000,
                 alpha: float = 0.05) -> "tuple[float, float]":
    """Percentile bootstrap CI for the mean of paired per-prompt deltas (mirrors
    agent/uplift._bootstrap_ci). n<2 -> degenerate (mean, mean)."""
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0)
    mean = sum(deltas) / n
    if n < 2:
        return (mean, mean)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return (lo, hi)


def _flags(condition: Condition, prompts: "tuple[str, ...]",
           scorer: Callable[[str], bool]) -> "list[int] | None":
    """Per-prompt {0,1} dangerous-behaviour flags. Returns None if the scorer
    raises on any completion (-> the run is UNMEASURED, fail-closed)."""
    out = []
    for p in prompts:
        try:
            out.append(1 if scorer(condition.generate(p)) else 0)
        except Exception:
            return None
    return out


def run_paired(task: EvalTask, control: Condition, treatment: Condition, *,
               seed: int = 0) -> PairedResult:
    """Score control & treatment on the same held-out bank; bootstrap the delta."""
    c = _flags(control, task.prompts, task.scorer)
    t = _flags(treatment, task.prompts, task.scorer)
    n = len(task.prompts)
    if c is None or t is None:
        # Fail-closed: scorer error => unmeasured, not a passing 0.0.
        return PairedResult(task.name, control.name, treatment.name, 0.0, 0.0, 0.0,
                            0.0, 0.0, False, n, seed, measured=False,
                            note="scorer raised; capability UNMEASURED (fail-closed)")
    cs = sum(c) / n if n else 0.0
    ts = sum(t) / n if n else 0.0
    deltas = [ti - ci for ci, ti in zip(c, t)]
    lo, hi = bootstrap_ci([float(d) for d in deltas], seed=seed)
    return PairedResult(task.name, control.name, treatment.name, cs, ts,
                        sum(deltas) / n if n else 0.0, lo, hi, lo > 0.0, n, seed,
                        measured=True)
