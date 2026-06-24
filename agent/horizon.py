# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Long-horizon autonomy curve — success rate vs task length.

`tools/run_long_horizon.py` logs ONE run's interventions; this measures the
*capability curve*: how reliably an agent completes tasks of increasing step
depth, judged by an EXTERNAL oracle (independent recomputation), never by the
agent itself. The honest headline number is the **effective horizon** — the
longest task length still solved at >=50% — which is how METR-style evals
summarise autonomy.

Tasks are chained-arithmetic: each step depends on the previous, so one slip
fails the whole task (the property that makes long horizons hard). Deterministic
and model-optional, so the harness is reproducible and offline-testable.
"""

from __future__ import annotations

import random
import re
from typing import Callable

DEFAULT_LENGTHS = (1, 2, 4, 8, 16, 32)


def make_chain_task(length: int, seed: int) -> dict:
    """A deterministic ``length``-step dependent-arithmetic task with a gold value."""
    rng = random.Random(seed)
    val = rng.randint(1, 9)
    desc = [f"Start with {val}."]
    for _ in range(length):
        op = rng.choice(["+", "*", "-"])
        k = rng.randint(1, 9)
        val = val + k if op == "+" else val * k if op == "*" else val - k
        desc.append(f"Then {op} {k}.")
    desc.append("What is the final value? Answer with just the number.")
    return {"length": length, "prompt": " ".join(desc), "gold": val}


def _final_int(text: str) -> "int | None":
    nums = re.findall(r"-?\d+", text or "")
    return int(nums[-1]) if nums else None


def perfect_solver(task: dict) -> str:
    return str(task["gold"])


def noisy_solver(per_step_error: float, *, seed: int = 0) -> Callable[[dict], str]:
    """A solver that gets each step right with prob ``1-per_step_error`` — so its
    success decays geometrically with length, the realistic long-horizon shape."""
    rng = random.Random(seed)

    def solve(task: dict) -> str:
        if rng.random() <= (1 - per_step_error) ** task["length"]:
            return str(task["gold"])
        return str(task["gold"] + rng.choice([-3, -2, -1, 1, 2, 3]))

    return solve


def model_solver(spec: str) -> Callable[[dict], str]:
    """Solve with a real model via the unified adapter (judged by the oracle)."""
    from agent.model import default_client

    client = default_client(spec)
    sys_prompt = "You are careful. Compute step by step and end with the final number only."
    return lambda task: getattr(client.generate(sys_prompt, task["prompt"]), "text", "") or ""


def horizon_curve(
    solve_fn: Callable[[dict], str], *,
    lengths=DEFAULT_LENGTHS, trials: int = 20, seed: int = 0,
) -> dict:
    """Run ``trials`` tasks per length; verify by exact-match of the final value
    against an independent recomputation. Returns the curve + effective horizon."""
    curve = []
    for n in lengths:
        succ = 0
        for t in range(trials):
            task = make_chain_task(n, seed=seed * 100003 + n * 101 + t)
            succ += int(_final_int(solve_fn(task)) == task["gold"])
        curve.append({"length": n, "trials": trials, "successRate": round(succ / trials, 4)})
    effective = max([c["length"] for c in curve if c["successRate"] >= 0.5], default=0)
    return {
        "curve": curve,
        "effectiveHorizon": effective,
        "oracle": "independent recomputation (exact-match final value) — never self-judged",
    }
