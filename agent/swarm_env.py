# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Swarm execution environment — the trainable world for a team-agent policy.

The pieces already exist separately: ``agent/swarm_router.py`` (the routing policy seam),
``agent/subagent.py`` (real isolated delegation with least-privilege tools + bounded budgets),
``agent/swarm_trust_boundary.py`` (the inter-agent gate), and ``provenance_bench/swarm_rl.py``
(the machine-verified reward). This module is the missing GLUE that composes them into one
steppable, verifiable episode — the prerequisite an RL loop needs to TRAIN a model to orchestrate:

    task --router.decide--> SwarmPlan --to_specs--> least-privilege subagents
         --run--> child outputs --TRUST BOUNDARY--> only gate-clean enter shared state
         --fail-closed reduce over ADMITTED only--> synthesis
         --machine verify--> SwarmOutcome --> reward

Why this and not ``agent.subagent.delegate`` alone: ``delegate`` reduces over children that are
merely harness-``ok`` (succeeded + on budget). A child can succeed yet hallucinate an attribution,
and that flows straight into the synthesis. Here the **trust boundary is the inter-agent contract**:
a child's output enters the shared state (and the reduce) ONLY if it clears the gate. The reward is
the same unhackable, machine-checked signal the router is trained against.

Seams (what a trainer overrides):
  * ``router`` — the policy. Default ``SwarmRouter`` (deterministic v1); the trained head overrides
    ``decide``. This is the surface RL optimises.
  * ``child_runner`` — how a plan's subagents actually execute. Default wraps
    ``agent.subagent.delegate`` (the real harness loop). Tests inject a deterministic runner so the
    environment CONTRACT (route -> gate -> reduce -> reward) is CI-checkable without a model.

Deterministic and offline with an injected ``child_runner``; the live runner needs the harness +
a model client (exercised in real runs, not CI). Makes no capability/AGI claim — it is the world,
not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.gate import check_response
from agent.swarm_router import SwarmPlan, SwarmRouter
from agent.swarm_trust_boundary import AgentMessage, GatedEntry, GatedSharedState
from provenance_bench.swarm_rl import (
    SwarmOutcome,
    TrajectoryOutcome,
    swarm_reward,
    trajectory_reward,
)

ABSTAIN = ("Insufficient verified basis: no subagent output cleared the provenance gate. "
           "Abstaining rather than synthesising from unverified work. Not advice.")


@dataclass
class ChildOutput:
    """One subagent's contribution, before the trust boundary decides if siblings may read it."""

    label: str
    agent_id: str
    text: str
    ok: bool = True            # harness-ok (succeeded + on budget) — necessary, NOT sufficient
    cost_usd: float = 0.0


@dataclass
class SwarmEpisode:
    """One full transition: plan -> gated execution -> reduce -> reward. The RL step."""

    task: str
    plan: SwarmPlan
    children: "list[ChildOutput]"
    admitted: "list[GatedEntry]"
    held: "list[GatedEntry]"
    synthesis: str
    outcome: SwarmOutcome
    reward: float = 0.0

    @property
    def n_failed_gate(self) -> int:
        return self.outcome.n_agents_failed_gate

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "mode": self.plan.mode,
            "nChildren": len(self.children),
            "admitted": [e.agent_id for e in self.admitted],
            "held": [{"agent": e.agent_id, "violations": e.violations} for e in self.held],
            "verifiedSuccess": self.outcome.verified_success,
            "nFailedGate": self.n_failed_gate,
            "reward": round(self.reward, 4),
            "synthesis": self.synthesis,
        }


def default_child_runner(plan: SwarmPlan, *, task: str, client=None) -> "list[ChildOutput]":
    """Execute a plan's subagents for real via the isolated-delegation harness.

    Imported lazily so the environment module stays importable (and the contract testable with an
    injected runner) even where the full harness / model client is unavailable."""
    from agent.subagent import SubagentSpec, delegate

    specs = plan.to_specs()
    if not specs:  # solo plan -> a single backbone child
        specs = [SubagentSpec(goal=task, label="solo", max_steps=3)]
    res = delegate(task, specs, client=client, synthesize=False)
    return [
        ChildOutput(label=c.spec.label or c.spec.goal[:40], agent_id=c.task_id,
                    text=c.final_text, ok=c.ok, cost_usd=c.cost_usd)
        for c in res.children
    ]


def _gated_reduce(admitted: "list[GatedEntry]") -> str:
    """Fail-closed synthesis over ADMITTED children only. A real reduce model can replace this
    concatenation; the contract is that HELD output is never an input."""
    if not admitted:
        return ABSTAIN
    return "\n\n".join(f"### {e.agent_id}\n{e.content}" for e in admitted)


def run_swarm_episode(
    task: str,
    *,
    router: "SwarmRouter | None" = None,
    child_runner=None,
    client=None,
    question: "str | None" = None,
    mode: str = "advisor",
) -> SwarmEpisode:
    """Step the environment once: route, execute, gate, reduce, reward. Deterministic given a
    deterministic ``child_runner`` (and the gate/verifiers, which are deterministic)."""
    router = router or SwarmRouter()
    plan = router.decide(task)

    runner = child_runner or (lambda p: default_child_runner(p, task=task, client=client))
    children = list(runner(plan))

    # The trust boundary: a child enters shared state (and the reduce) only if it clears the gate.
    state = GatedSharedState()
    n_failed_gate = 0
    for ch in children:
        if not ch.ok:
            n_failed_gate += 1
            continue
        entry = state.submit(AgentMessage(agent_id=ch.agent_id, content=ch.text,
                                          question=question or task, mode=mode))
        if not entry.admitted:
            n_failed_gate += 1
    admitted, held = state.readable(), state.held()

    synthesis = _gated_reduce(admitted)
    syn = check_response(synthesis, mode=mode, question=question or task, route_claims=True)
    # Verified success: at least one child cleared the gate AND the synthesis itself is gate-clean.
    verified_success = 1.0 if (admitted and not (syn.get("violations") or [])) else 0.0

    outcome = SwarmOutcome(plan=plan, verified_success=verified_success,
                           n_agents_failed_gate=min(n_failed_gate, max(plan.n_agents, len(children), 1)),
                           serial_depth=1)
    reward = swarm_reward(outcome)
    return SwarmEpisode(task=task, plan=plan, children=children, admitted=admitted, held=held,
                        synthesis=synthesis, outcome=outcome, reward=reward)


@dataclass
class SwarmTrajectory:
    """A multi-turn rollout: a sequence of swarm episodes toward one goal, scored by the
    multi-turn ``trajectory_reward`` (KL-controlled, length-normalised). This is the unit a
    GRPO/PPO loop trains the orchestration policy on."""

    goal: str
    episodes: "list[SwarmEpisode]"
    outcome: TrajectoryOutcome
    reward: float = 0.0

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "nTurns": len(self.episodes),
            "finalVerifiedSuccess": self.outcome.final_verified_success,
            "totalGateFailures": self.outcome.total_gate_failures,
            "reward": round(self.reward, 4),
            "turns": [ep.to_dict() for ep in self.episodes],
        }


def run_swarm_trajectory(
    turns: "list[str]",
    *,
    goal: "str | None" = None,
    router: "SwarmRouter | None" = None,
    child_runner=None,
    client=None,
    question: "str | None" = None,
    mode: str = "advisor",
    kl_per_turn: "tuple[float, ...]" = (),
) -> SwarmTrajectory:
    """Run a sequence of sub-task ``turns`` as swarm episodes and fold them into one
    multi-turn ``TrajectoryOutcome``. ``final_verified_success`` is the LAST turn's verified
    success (did the trajectory finish gate-clean); ``kl_per_turn`` is supplied by the trainer
    (KL vs the reference policy) — empty offline. Deterministic given a deterministic runner."""
    episodes = [
        run_swarm_episode(t, router=router, child_runner=child_runner, client=client,
                          question=question, mode=mode)
        for t in turns
    ]
    outcome = TrajectoryOutcome(
        turns=[ep.outcome for ep in episodes],
        final_verified_success=(episodes[-1].outcome.verified_success if episodes else 0.0),
        kl_per_turn=tuple(kl_per_turn),
    )
    reward = trajectory_reward(outcome)
    return SwarmTrajectory(goal=goal or (turns[0] if turns else ""), episodes=episodes,
                           outcome=outcome, reward=reward)


# --- deterministic contract fixtures (no harness, no model) ------------------------------------
def _stub_runner(outputs: "list[ChildOutput]"):
    return lambda _plan: outputs


def offline_invariants() -> "tuple[bool, dict]":
    """Falsifiable, deterministic invariants for the environment CONTRACT (injected runner)."""
    task = "Did Confucius write the Dao De Jing? Cite sources."
    clean = ChildOutput("researcher", "ag.clean",
                        "No — Confucius did not write the Dao De Jing; it is a distinct Daoist text "
                        "traditionally attributed to Laozi. The Daoist and Confucian traditions are "
                        "separate. 來源存疑。")
    poison = ChildOutput("rogue", "ag.poison", "Confucius wrote the Dao De Jing, unifying both traditions.")

    checks: dict[str, bool] = {}

    ep_clean = run_swarm_episode(task, child_runner=_stub_runner([clean]), question=task)
    checks["clean_admitted"] = [e.agent_id for e in ep_clean.admitted] == ["ag.clean"]
    checks["clean_success"] = ep_clean.outcome.verified_success == 1.0

    ep_mixed = run_swarm_episode(task, child_runner=_stub_runner([clean, poison]), question=task)
    checks["poison_held"] = "ag.poison" in [e.agent_id for e in ep_mixed.held]
    checks["poison_not_in_synthesis"] = "ag.poison" not in ep_mixed.synthesis
    checks["mixed_counts_failed_gate"] = ep_mixed.n_failed_gate >= 1

    ep_allbad = run_swarm_episode(task, child_runner=_stub_runner([poison]), question=task)
    checks["all_poison_abstains"] = ep_allbad.synthesis == ABSTAIN
    checks["all_poison_zero_success"] = ep_allbad.outcome.verified_success == 0.0

    # A clean episode out-rewards a poisoned one (the policy gradient the trainer would follow).
    checks["clean_outrewards_poison"] = ep_clean.reward > ep_allbad.reward

    # Determinism: same inputs -> same reward.
    again = run_swarm_episode(task, child_runner=_stub_runner([clean]), question=task)
    checks["deterministic"] = again.reward == ep_clean.reward

    # Multi-turn trajectory: a 3-turn all-clean rollout finishes gate-clean and out-rewards a
    # rollout whose final turn is poisoned (the final-success signal the trainer follows).
    turns = [task, task, task]
    traj_clean = run_swarm_trajectory(turns, child_runner=_stub_runner([clean]), question=task)
    traj_fail = run_swarm_trajectory(turns, child_runner=_stub_runner([poison]), question=task)
    checks["traj_clean_final_success"] = traj_clean.outcome.final_verified_success == 1.0
    checks["traj_fail_final_zero"] = traj_fail.outcome.final_verified_success == 0.0
    checks["traj_clean_outrewards_fail"] = traj_clean.reward > traj_fail.reward
    checks["traj_kl_penalised"] = (
        run_swarm_trajectory(turns, child_runner=_stub_runner([clean]), question=task,
                             kl_per_turn=(2.0, 2.0, 2.0)).reward < traj_clean.reward
    )

    ok = all(checks.values())
    return ok, {"checks": checks, "cleanReward": round(ep_clean.reward, 3),
                "poisonReward": round(ep_allbad.reward, 3),
                "trajClean": round(traj_clean.reward, 3), "trajFail": round(traj_fail.reward, 3)}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ok, detail = offline_invariants()
    print("Swarm execution environment invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  clean/poison reward:", detail["cleanReward"], "/", detail["poisonReward"])
    raise SystemExit(0 if ok else 1)
