# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Solo-vs-Swarm benchmark — does fanning out a team actually beat the backbone alone,
and at what cost?

This is the **no-overclaim gate** for the Agentic-MoE design
(``docs/11-Platform/Agentic-MoE-Swarm.md`` §7). A swarm "win" only counts if verified
task success beats the **solo same-backbone baseline** AND the cost delta is reported,
never hidden. It is deliberately solver-agnostic: you pass two callables

    solve_solo(task)           -> answer text
    solve_swarm(task, plan)    -> answer text     (plan from agent.swarm_router)

and a deterministic ``verify(answer, gold) -> bool`` referee that is **disjoint** from
the policy (same independence principle as ``provenance_bench/judge.py`` vs the gate).
The harness scores both arms, bootstraps a paired CI on the success delta, and sums the
honest compute cost (router-estimated steps) of each arm.

Offline + deterministic: ``offline_invariants()`` drives the whole thing with synthetic
solvers where the swarm helps on hard tasks and ties on easy ones, proving the harness
math (delta>0 with CI excluding zero, cost up) and the no-overclaim guard (a swarm that
*doesn't* help yields a CI that includes zero → no win).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agent.swarm_router import SwarmPlan, SwarmRouter

# Solver/referee signatures.
SolveSolo = Callable[[str], str]
SolveSwarm = Callable[[str, SwarmPlan], str]
Verify = Callable[[str, str], bool]


@dataclass
class Task:
    task: str
    gold: str
    hard: bool = False  # metadata only (used by synthetic invariants); the router decides routing


@dataclass
class ArmResult:
    arm: str
    n: int
    n_correct: int
    cost_steps: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "n": self.n,
            "nCorrect": self.n_correct,
            "accuracy": round(self.accuracy, 4),
            "costSteps": self.cost_steps,
        }


@dataclass
class SwarmReport:
    solo: ArmResult
    swarm: ArmResult
    delta: float
    ci95: "tuple[float, float]"
    cost_ratio: float
    per_task: list[dict] = field(default_factory=list)

    @property
    def is_win(self) -> bool:
        """No-overclaim: a win requires the paired delta CI to exclude zero."""
        return self.ci95[0] > 0.0

    def to_dict(self) -> dict:
        return {
            "solo": self.solo.to_dict(),
            "swarm": self.swarm.to_dict(),
            "deltaAccuracy": round(self.delta, 4),
            "ci95": [round(self.ci95[0], 4), round(self.ci95[1], 4)],
            "ciExcludesZero": self.is_win,
            "costRatioSwarmOverSolo": round(self.cost_ratio, 3),
            "verdict": (
                "swarm_wins" if self.is_win else "no_swarm_advantage_at_this_cost"
            ),
            "note": "Swarm 'win' = verified success delta CI excludes zero AND cost reported (not hidden).",
        }


def _paired_bootstrap_ci(
    solo_hits: "list[int]", swarm_hits: "list[int]", *, iters: int = 2000, seed: int = 0
) -> "tuple[float, float]":
    """Deterministic paired bootstrap on the per-task (swarm-solo) difference. Uses a
    fixed LCG so the CI is reproducible with no numpy dependency."""
    diffs = [s - o for s, o in zip(swarm_hits, solo_hits)]
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    state = (seed * 2_654_435_761 + 1) & 0xFFFFFFFF
    means: list[float] = []
    for _ in range(iters):
        total = 0
        for _ in range(n):
            state = (1_103_515_245 * state + 12_345) & 0x7FFFFFFF
            total += diffs[state % n]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * iters)]
    hi = means[min(int(0.975 * iters), iters - 1)]
    return (lo, hi)


def run_benchmark(
    tasks: "list[Task]",
    solve_solo: SolveSolo,
    solve_swarm: SolveSwarm,
    verify: Verify,
    *,
    router: SwarmRouter | None = None,
) -> SwarmReport:
    """Score the solo backbone vs the routed swarm on ``tasks`` and report the gated delta."""
    router = router or SwarmRouter()
    solo_hits: list[int] = []
    swarm_hits: list[int] = []
    solo_cost = 0
    swarm_cost = 0
    per_task: list[dict] = []

    for t in tasks:
        plan = router.decide(t.task)
        a_solo = solve_solo(t.task)
        a_swarm = solve_swarm(t.task, plan)
        ok_solo = bool(verify(a_solo, t.gold))
        ok_swarm = bool(verify(a_swarm, t.gold))
        solo_hits.append(int(ok_solo))
        swarm_hits.append(int(ok_swarm))
        solo_cost += 3  # the backbone's own steps (matches SwarmPlan solo estimate)
        swarm_cost += plan.est_cost_steps
        per_task.append({
            "task": t.task[:60],
            "mode": plan.mode,
            "soloOk": ok_solo,
            "swarmOk": ok_swarm,
            "planCostSteps": plan.est_cost_steps,
        })

    n = len(tasks)
    solo = ArmResult("solo", n, sum(solo_hits), solo_cost)
    swarm = ArmResult("swarm", n, sum(swarm_hits), swarm_cost)
    delta = swarm.accuracy - solo.accuracy
    ci = _paired_bootstrap_ci(solo_hits, swarm_hits)
    cost_ratio = (swarm_cost / solo_cost) if solo_cost else float("inf")
    return SwarmReport(solo, swarm, delta, ci, cost_ratio, per_task)


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # A synthetic world: 'hard' tasks need the swarm; 'easy' tasks both get right.
    # gold == "SWARM" for hard tasks (only the swarm arm produces it), "EASY" otherwise.
    tasks = (
        [Task(f"Compare disputed claim {i} versus its rival, citing primary sources", "SWARM", hard=True)
         for i in range(12)]
        + [Task(f"What is term {i}", "EASY", hard=False) for i in range(12)]
    )

    def verify(ans: str, gold: str) -> bool:
        return ans.strip() == gold

    # Solo solver: gets easy right, misses hard (answers "SOLO").
    def solo(task: str) -> str:
        return "EASY" if task.lower().startswith("what is") else "SOLO"

    # Swarm solver: gets easy right AND hard right when the router actually fanned out.
    def make_swarm_solver(helps: bool):
        def swarm(task: str, plan: SwarmPlan) -> str:
            if task.lower().startswith("what is"):
                return "EASY"
            # Only earn the hard answer if (a) the router fanned out and (b) this world helps.
            if plan.mode == "swarm" and helps:
                return "SWARM"
            return "SOLO"
        return swarm

    rep = run_benchmark(tasks, solo, make_swarm_solver(True), verify)

    # 1. Swarm beats solo on this world.
    checks["swarm_beats_solo"] = rep.swarm.accuracy > rep.solo.accuracy
    detail["soloAcc"] = round(rep.solo.accuracy, 3)
    detail["swarmAcc"] = round(rep.swarm.accuracy, 3)

    # 2. The win is real: paired CI excludes zero.
    checks["ci_excludes_zero_when_real"] = rep.is_win
    detail["ci95"] = [round(rep.ci95[0], 3), round(rep.ci95[1], 3)]

    # 3. Cost is reported and higher for the swarm (no hidden cost).
    checks["cost_reported_and_higher"] = rep.cost_ratio > 1.0
    detail["costRatio"] = round(rep.cost_ratio, 3)

    # 4. No-overclaim guard: a swarm that DOESN'T help yields a CI including zero -> no win.
    rep_null = run_benchmark(tasks, solo, make_swarm_solver(False), verify)
    checks["no_false_win_when_useless"] = (not rep_null.is_win) and rep_null.delta == 0.0

    # 5. Determinism: same inputs → identical report dict.
    rep2 = run_benchmark(tasks, solo, make_swarm_solver(True), verify)
    checks["deterministic"] = rep.to_dict() == rep2.to_dict()

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Swarm benchmark offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  solo/swarm acc:", detail.get("soloAcc"), "/", detail.get("swarmAcc"),
          " CI:", detail.get("ci95"), " costRatio:", detail.get("costRatio"))
    raise SystemExit(0 if ok else 1)
