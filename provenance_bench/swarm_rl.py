# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Swarm-Router RLVR reward — the Stage-3 training signal of the Agentic-MoE design.

``docs/11-Platform/Agentic-MoE-Swarm.md`` §5 Stage 3 defines the reward the trained
Swarm-Router is optimised against:

    R = verified_success
      − λ_cost  · compute_spent          (tokens × agents × steps)
      − λ_lb    · load_imbalance         (Switch aux loss, lifted to teams)
      − λ_trust · over_reliance          (picked a team whose output failed the gate)
      − λ_lat   · wall_clock             (needless serial depth)

The thing that lets Sophia train this **honestly** is that ``verified_success`` and
``over_reliance`` are *machine-checked*, not a learnable reward model — so the reward is
**deterministic and unhackable** (the same property ``provenance_bench/governed_rl.py``
relies on). A router can't game a verifier it doesn't control; it can only learn to
*route to the teams whose work survives the verifier* at the lowest cost.

This module is the **reward + its offline invariants only** — pure-Python, no torch, no
GPU, CI-testable. The GPU policy-gradient loop (GRPO/PPO) that consumes
:func:`make_grpo_reward` is the guarded glue in ``training/swarm_router/train_grpo.py``.

Load imbalance replicates the exact Switch formula (Fedus et al. 2021, eq. 4) used by
``moe/router.load_balancing_loss``, in pure Python (no numpy dependency), so the penalty
is the same load-balancing *objective* that balances an in-weights MoE — just lifted from
tokens-over-FFN-experts to tasks-over-agent-teams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.swarm_router import TEAMS, SwarmPlan

# Default penalty weights. Tuned so a clean solo answer and a justified swarm both score
# near the success ceiling, while a wasteful or gate-failing swarm is pushed below solo.
LAMBDA_COST = 0.015     # per compute-step
LAMBDA_LB = 0.10        # per unit of (aux - 1.0) imbalance
LAMBDA_TRUST = 0.40     # per fraction of dispatched agents that failed the gate
LAMBDA_LAT = 0.02       # per unit of serial depth beyond 1
LAMBDA_KL = 0.10        # per unit of mean KL(policy‖reference) — multi-turn stability control

REWARD_FLOOR = -1.0
REWARD_CEIL = 1.0


@dataclass
class SwarmOutcome:
    """Everything the (machine-checked) reward needs about one executed plan. All fields
    are produced by deterministic checkers, never by a learnable judge."""

    plan: SwarmPlan
    verified_success: float          # in [0,1] from agent.gate / math_verify / etc. (machine-checked)
    n_agents_failed_gate: int = 0    # children whose output failed the fail-closed gate
    serial_depth: int = 1            # longest dependent chain (1 = fully parallel)

    @property
    def compute_steps(self) -> int:
        return self.plan.est_cost_steps

    @property
    def n_agents(self) -> int:
        return max(self.plan.n_agents, 1)

    @property
    def over_reliance(self) -> float:
        """Fraction of dispatched agents whose work failed the verifier — the signal the
        trust-balance penalty kills. A router that learns to lean on a cheap team that
        *looks* confident but fails the gate is punished here."""
        return self.n_agents_failed_gate / self.n_agents


def team_load_imbalance(plans: "list[SwarmPlan]", *, num_teams: int | None = None) -> float:
    """Switch-Transformer load-balancing aux (Fedus et al. 2021, eq. 4) lifted from
    tokens-over-FFN-experts to tasks-over-agent-teams: aux = E · Σ_e f_e · P_e, where
    f_e is the fraction of *dispatched agents* going to team e and P_e is the mean
    *routing mass* on e. Uniform routing → 1.0; collapse onto one team → ~E. Minimising
    it spreads load, exactly as in moe/router.py."""
    team_names = list(TEAMS)
    E = num_teams or len(team_names)
    counts = {t: 0 for t in team_names}
    mass = {t: 0.0 for t in team_names}
    total_agents = 0
    n_plans = 0
    for p in plans:
        if p.mode != "swarm" or not p.assignments:
            continue
        n_plans += 1
        plan_total = sum(a.k for a in p.assignments)
        for a in p.assignments:
            counts[a.team] += a.k
            total_agents += a.k
            # routing "probability" mass for this plan = share of this plan's agents.
            mass[a.team] += (a.k / plan_total) if plan_total else 0.0
    if total_agents == 0 or n_plans == 0:
        return 1.0  # nothing dispatched → defined as the balanced floor
    aux = 0.0
    for t in team_names:
        f_e = counts[t] / total_agents
        p_e = mass[t] / n_plans
        aux += f_e * p_e
    return E * aux


def swarm_reward(
    outcome: SwarmOutcome,
    *,
    lambda_cost: float = LAMBDA_COST,
    lambda_trust: float = LAMBDA_TRUST,
    lambda_lat: float = LAMBDA_LAT,
    load_imbalance: float = 1.0,
    lambda_lb: float = LAMBDA_LB,
) -> float:
    """The bounded, deterministic per-trajectory reward. ``load_imbalance`` is a
    batch-level quantity (from :func:`team_load_imbalance`); pass 1.0 for the
    per-sample reward when balance is handled at the batch level."""
    r = float(outcome.verified_success)
    r -= lambda_cost * outcome.compute_steps
    r -= lambda_trust * outcome.over_reliance
    r -= lambda_lat * max(outcome.serial_depth - 1, 0)
    r -= lambda_lb * max(load_imbalance - 1.0, 0.0)
    return max(REWARD_FLOOR, min(REWARD_CEIL, r))


@dataclass
class TrajectoryOutcome:
    """A multi-turn agentic rollout: a sequence of executed plans (turns), the machine-checked
    success of the WHOLE trajectory, and optional per-turn KL(policy‖reference) supplied by the
    trainer. The single-turn :class:`SwarmOutcome` is the degenerate one-turn case.

    Everything the reward reads is produced by deterministic checkers — ``final_verified_success``
    by the gate/verifiers, ``kl_per_turn`` by the trainer's reference-policy comparison — so the
    multi-turn reward keeps the same unhackable property as the single-turn one."""

    turns: "list[SwarmOutcome]"
    final_verified_success: float            # in [0,1] from agent.gate over the FINAL trajectory
    kl_per_turn: "tuple[float, ...]" = ()    # KL(policy‖ref) per turn (>=0), from the trainer

    @property
    def n_turns(self) -> int:
        return max(len(self.turns), 1)

    @property
    def total_compute_steps(self) -> int:
        return sum(t.compute_steps for t in self.turns)

    @property
    def total_agents(self) -> int:
        return sum(t.n_agents for t in self.turns)

    @property
    def total_gate_failures(self) -> int:
        return sum(t.n_agents_failed_gate for t in self.turns)

    @property
    def mean_kl(self) -> float:
        return (sum(self.kl_per_turn) / len(self.kl_per_turn)) if self.kl_per_turn else 0.0


def trajectory_reward(
    traj: TrajectoryOutcome,
    *,
    lambda_cost: float = LAMBDA_COST,
    lambda_trust: float = LAMBDA_TRUST,
    lambda_kl: float = LAMBDA_KL,
    length_normalize: bool = True,
) -> float:
    """Bounded, deterministic reward for a multi-turn tool-use rollout::

        R = final_verified_success
          − λ_cost  · (total_compute_steps / n_turns if length_normalize else total)
          − λ_trust · over_reliance            (gate failures across the whole trajectory)
          − λ_kl    · mean_kl                  (KL control vs the reference policy)

    The two fixes the positioning plan calls for so GRPO does not collapse on long rollouts:
    **length normalisation** (cost divided by turns, so a longer trajectory reaching the same
    verified success is not penalised beyond its per-turn cost) and **KL control** (an explicit
    penalty on divergence from the reference policy). Reuses the single-turn over-reliance signal,
    summed across turns. Bounded to ``[REWARD_FLOOR, REWARD_CEIL]`` like :func:`swarm_reward`."""
    r = float(traj.final_verified_success)
    cost = traj.total_compute_steps / (traj.n_turns if length_normalize else 1)
    r -= lambda_cost * cost
    over_reliance = traj.total_gate_failures / max(traj.total_agents, 1)
    r -= lambda_trust * over_reliance
    r -= lambda_kl * max(traj.mean_kl, 0.0)
    return max(REWARD_FLOOR, min(REWARD_CEIL, r))


def trajectory_invariants() -> "tuple[bool, dict]":
    """Falsifiable, deterministic invariants for the multi-turn reward (no model, no network)."""
    from agent.swarm_router import SwarmRouter

    r = SwarmRouter()
    step = r.decide("Compare disputed authorship of the Dao De Jing versus the Analects, citing sources")
    turns = [SwarmOutcome(step, verified_success=1.0) for _ in range(3)]

    checks: dict[str, bool] = {}
    detail: dict = {}

    win = trajectory_reward(TrajectoryOutcome(turns, final_verified_success=1.0))
    lose = trajectory_reward(TrajectoryOutcome(turns, final_verified_success=0.0))
    checks["success_raises_reward"] = win > lose

    # KL control: a trajectory that drifts from the reference policy scores lower.
    calm = trajectory_reward(TrajectoryOutcome(turns, 1.0, kl_per_turn=(0.0, 0.0, 0.0)))
    drift = trajectory_reward(TrajectoryOutcome(turns, 1.0, kl_per_turn=(2.0, 2.0, 2.0)))
    checks["kl_control_penalises_drift"] = calm > drift
    detail["calm"] = round(calm, 3)
    detail["drift"] = round(drift, 3)

    # Over-reliance across turns: a trajectory whose agents kept failing the gate scores lower.
    clean = trajectory_reward(TrajectoryOutcome(turns, 1.0))
    leaky_turns = [SwarmOutcome(step, verified_success=1.0, n_agents_failed_gate=step.n_agents)
                   for _ in range(3)]
    leaky = trajectory_reward(TrajectoryOutcome(leaky_turns, 1.0))
    checks["over_reliance_penalised"] = clean > leaky

    # Length normalisation stops long-rollout cost-collapse: with normalisation a 3-turn
    # trajectory is penalised less than without it, all else equal.
    normed = trajectory_reward(TrajectoryOutcome(turns, 1.0), length_normalize=True)
    raw = trajectory_reward(TrajectoryOutcome(turns, 1.0), length_normalize=False)
    checks["length_normalization_reduces_cost_penalty"] = normed > raw

    # Bounded even under a pathological long, drifting, gate-failing trajectory.
    worst = trajectory_reward(TrajectoryOutcome(
        [SwarmOutcome(step, verified_success=0.0, n_agents_failed_gate=step.n_agents) for _ in range(20)],
        final_verified_success=0.0, kl_per_turn=tuple(50.0 for _ in range(20))))
    checks["bounded"] = REWARD_FLOOR <= worst <= REWARD_CEIL

    # Determinism: identical trajectory → identical reward.
    a = trajectory_reward(TrajectoryOutcome(turns, 0.7, kl_per_turn=(0.1, 0.2, 0.3)))
    b = trajectory_reward(TrajectoryOutcome(turns, 0.7, kl_per_turn=(0.1, 0.2, 0.3)))
    checks["deterministic"] = a == b

    # Degenerate single-turn trajectory is well-defined (no div-by-zero on empty turns).
    checks["empty_trajectory_defined"] = (
        REWARD_FLOOR <= trajectory_reward(TrajectoryOutcome([], 1.0)) <= REWARD_CEIL
    )

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


def make_grpo_reward(
    *,
    score_success: Callable[[Any, SwarmPlan], float],
    count_gate_failures: Callable[[Any, SwarmPlan], int] | None = None,
) -> Callable:
    """Build a GRPO-style reward callable ``f(completions, plans=..., **kw) -> list[float]``
    for a TRL/verl-style trainer. ``score_success`` and ``count_gate_failures`` are the
    *machine* checkers (gate / verifier) — the trainer supplies them; this module never
    embeds a learnable judge, which is what keeps the reward unhackable."""

    def reward_fn(completions, plans: "list[SwarmPlan]", serial_depths: "list[int] | None" = None, **_) -> "list[float]":
        depths = serial_depths or [1] * len(plans)
        lb = team_load_imbalance(plans)
        out: list[float] = []
        for comp, plan, depth in zip(completions, plans, depths):
            success = float(score_success(comp, plan))
            failed = int(count_gate_failures(comp, plan)) if count_gate_failures else 0
            outcome = SwarmOutcome(plan=plan, verified_success=success,
                                   n_agents_failed_gate=failed, serial_depth=depth)
            out.append(swarm_reward(outcome, load_imbalance=lb))
        return out

    return reward_fn


def offline_invariants() -> "tuple[bool, dict]":
    from agent.swarm_router import SwarmRouter

    r = SwarmRouter()
    checks: dict[str, bool] = {}
    detail: dict = {}

    hard_plan = r.decide(
        "Compare the disputed authorship of the Dao De Jing versus the Analects, citing sources"
    )
    easy_plan = r.decide("hi")  # solo

    # 1. Success raises reward, all else equal.
    win = swarm_reward(SwarmOutcome(hard_plan, verified_success=1.0))
    lose = swarm_reward(SwarmOutcome(hard_plan, verified_success=0.0))
    checks["success_raises_reward"] = win > lose
    detail["win"] = round(win, 3)
    detail["lose"] = round(lose, 3)

    # 2. Cost penalty: a more expensive plan with equal success scores lower.
    cheap = swarm_reward(SwarmOutcome(easy_plan, verified_success=1.0))
    pricey = swarm_reward(SwarmOutcome(hard_plan, verified_success=1.0))
    checks["cost_penalised"] = cheap > pricey
    detail["soloCostSteps"] = easy_plan.est_cost_steps
    detail["swarmCostSteps"] = hard_plan.est_cost_steps

    # 3. Over-reliance (a dispatched team failed the gate) is penalised.
    clean = swarm_reward(SwarmOutcome(hard_plan, verified_success=1.0, n_agents_failed_gate=0))
    leaky = swarm_reward(SwarmOutcome(hard_plan, verified_success=1.0,
                                      n_agents_failed_gate=hard_plan.n_agents))
    checks["over_reliance_penalised"] = clean > leaky

    # 4. Latency: serial depth beyond 1 costs.
    parallel = swarm_reward(SwarmOutcome(hard_plan, verified_success=1.0, serial_depth=1))
    serial = swarm_reward(SwarmOutcome(hard_plan, verified_success=1.0, serial_depth=4))
    checks["latency_penalised"] = parallel > serial

    # 5. Load imbalance: a balanced batch (aux≈1) out-rewards a collapsed batch (aux>1).
    balanced_batch = [
        r.decide("Calculate 12*13 and verify"),
        r.decide("Which quote is misattributed to Einstein?"),
        r.decide("Compare Kant and Hume on causation citing sources"),
    ]
    collapsed_batch = [hard_plan, hard_plan, hard_plan]  # same teams every time → skew
    lb_bal = team_load_imbalance(balanced_batch)
    lb_col = team_load_imbalance(collapsed_batch)
    checks["imbalance_penalises_collapse"] = lb_col >= lb_bal
    detail["lbBalanced"] = round(lb_bal, 3)
    detail["lbCollapsed"] = round(lb_col, 3)

    # 6. Bounded in [-1, 1] even under a pathological plan.
    worst = swarm_reward(SwarmOutcome(hard_plan, verified_success=0.0,
                                      n_agents_failed_gate=hard_plan.n_agents, serial_depth=20),
                         load_imbalance=len(TEAMS))
    checks["bounded"] = REWARD_FLOOR <= worst <= REWARD_CEIL

    # 7. Determinism / unhackability: identical outcome → identical reward.
    checks["deterministic"] = (
        swarm_reward(SwarmOutcome(hard_plan, verified_success=0.7)) ==
        swarm_reward(SwarmOutcome(hard_plan, verified_success=0.7))
    )

    # 8. A clean solo answer beats a wasteful swarm that fails the gate (the policy the
    #    router should learn: don't fan out unless it survives verification).
    checks["solo_beats_wasteful_swarm"] = cheap > leaky

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Swarm RLVR reward offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  win/lose:", detail.get("win"), "/", detail.get("lose"),
          " lb bal/collapsed:", detail.get("lbBalanced"), "/", detail.get("lbCollapsed"))

    tok, tdetail = trajectory_invariants()
    print("Multi-turn trajectory reward invariants:", "PASS" if tok else "FAIL")
    for k, v in tdetail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  calm/drift:", tdetail.get("calm"), "/", tdetail.get("drift"))

    raise SystemExit(0 if (ok and tok) else 1)
