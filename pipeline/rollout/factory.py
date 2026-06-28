# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cache-stable rollout factory — the RLVR data pump (Phase 1).

Generates verifier-scored reasoning traces cheaply, porting the two ideas worth
porting from DeepSeek-Reasonix:

  1. **Append-only, prefix-stable sessions** (``session.Session``) so a
     prefix-caching provider re-bills only the new tail each turn.
  2. **Planner / Executor split in SEPARATE sessions.** A planner emits a
     structured plan; an executor carries it out in its *own* session. The two
     prefixes never mix, so neither is disturbed by the other's turns — both stay
     cache-warm. (Reasonix's key composition trick.)

Each rollout is scored by a *verifiable* reward (``provenance_bench`` math / code /
physics — judge-free ground truth) and emitted as an ``AgentTrajectory`` record
(validated by ``pretraining.vertical_data.schemas``). The point is throughput:
cheaper traces → more RLVR steps per dollar → the live GPU run becomes affordable.

Offline-first: with the ``mock`` backend (or the in-module ``ScriptedClient``) the
whole pipeline is deterministic and runs in CI with no network and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pipeline.rollout.cost import CostMeter
from pipeline.rollout.session import Session, count_tokens
from pipeline.rollout.tools import TOOL_RE, Tool

# A reward fn matches provenance_bench.*_reward.reward_for_problem:
#   (answer, gold) -> (score, detail)   with score in {-1.0, +1.0}
RewardFn = Callable[[str, str], "tuple[float, dict]"]

PLANNER_SYSTEM = (
    "You are the PLANNER. Produce a short, numbered plan to solve the problem. "
    "Do not compute the final answer; hand the plan to the executor."
)
EXECUTOR_SYSTEM = (
    "You are the EXECUTOR. Follow the given plan, show the working, and put the "
    "final answer in \\boxed{} with units where applicable."
)


class ScriptedClient:
    """Duck-typed stand-in for ``agent.model.ModelClient`` for tests/invariants.

    ``generate(system, user)`` returns a scripted reply keyed by role (detected from
    the system prompt). The executor side can be a *sequence* (``answers=[...]``) to
    drive a multi-step tool loop deterministically — e.g. emit ``TOOL: calc(...)``
    then a final ``\\boxed{}``. Deterministic; no live model.
    """

    def __init__(self, *, plan: str = "1. identify formula\n2. substitute\n3. solve",
                 answer: str = "", answers: "list[str] | None" = None) -> None:
        self.plan = plan
        self._answers = list(answers) if answers is not None else [answer]
        self.calls = 0
        self._exec_idx = 0

    def generate(self, system: str, user: str, **_: Any) -> Any:
        self.calls += 1
        if "PLANNER" in system:
            return _Result(self.plan)
        i = min(self._exec_idx, len(self._answers) - 1)
        self._exec_idx += 1
        return _Result(self._answers[i])


@dataclass
class _Result:
    text: str

    @property
    def prompt_tokens(self) -> int:
        return 0

    @property
    def completion_tokens(self) -> int:
        return count_tokens(self.text)


class RolloutFactory:
    """Generate verifier-scored traces with cache-stable planner/executor sessions."""

    def __init__(self, client: Any = None, *, context_window: int = 8192,
                 compact_ratio: float = 0.8, cache_rate: float = 0.1) -> None:
        if client is None:
            from agent.model import default_client
            client = default_client("mock")
        self.client = client
        self.context_window = context_window
        self.compact_ratio = compact_ratio
        self.cache_rate = cache_rate

    # -- one cache-stable model turn ---------------------------------------- #
    def _turn(self, session: Session, user: str, meter: CostMeter) -> str:
        """Append a user message, generate, append the reply — accounting cache cost.

        The prior prefix (everything already in the session) is a cache hit; only
        the new user message + completion are fresh. Compacts once when the prefix
        crosses ``compact_ratio`` of the window (low-frequency, cache-respecting).
        """
        if session.needs_compaction():
            session.compact(summary=_summarize(session))
        cached_prefix = session.cached_prefix_tokens
        fresh_input = count_tokens(user)
        session.append("user", user)
        result = self.client.generate(session.system, _render(session))
        reply = getattr(result, "text", "") or ""
        session.append("assistant", reply)
        session.mark_sent()
        meter.record_turn(
            cached_prefix_tokens=cached_prefix,
            fresh_input_tokens=fresh_input,
            completion_tokens=count_tokens(reply),
        )
        return reply

    # -- a single planner->executor rollout --------------------------------- #
    def rollout(self, goal: str, *, gold: str | None = None,
                reward_for: RewardFn | None = None,
                tools: "dict[str, Tool] | None" = None, max_executor_steps: int = 4,
                source: str = "rollout-factory", license: str = "Apache-2.0") -> dict:
        """Run one planner→executor rollout and return a scored ``AgentTrajectory``.

        Planner and executor get SEPARATE sessions (disjoint prefixes). The executor
        runs a tool loop: while its reply emits ``TOOL: name(arg)`` and steps remain,
        the tool's observation is appended and the executor continues — each turn
        deepening the (cache-warm) session. The loop ends on a reply with no tool
        call (the final answer) or at ``max_executor_steps``. The reward is the
        verifiable ``reward_for(answer, gold)`` when supplied (else null).
        """
        tools = tools or {}
        meter = CostMeter(cache_rate=self.cache_rate)
        planner = Session(system=PLANNER_SYSTEM, context_window=self.context_window,
                          compact_ratio=self.compact_ratio)
        executor = Session(system=EXECUTOR_SYSTEM, context_window=self.context_window,
                           compact_ratio=self.compact_ratio)

        plan = self._turn(planner, f"Problem:\n{goal}", meter)
        steps: list[dict] = [{"action": "plan", "observation": plan}]

        reply = self._turn(executor, f"Plan:\n{plan}\n\nProblem:\n{goal}", meter)
        answer = reply
        for _ in range(max(0, max_executor_steps - 1)):
            m = TOOL_RE.search(reply)
            if not m:
                break  # no tool call -> this reply is the final answer
            name, arg = m.group(1), m.group(2)
            tool = tools.get(name)
            obs = tool(arg) if tool else f"error: unknown tool {name!r}"
            steps.append({"action": f"tool:{name}({arg})", "observation": obs})
            reply = self._turn(executor, f"Observation: {obs}\nContinue.", meter)
            answer = reply
        steps.append({"action": "execute", "observation": answer})

        reward: float | None = None
        verdict_detail: dict = {}
        if reward_for is not None and gold is not None:
            score, verdict_detail = reward_for(answer, gold)
            reward = 1.0 if score > 0 else 0.0

        return {
            "goal": goal,
            "steps": steps,
            "outcome": answer,
            "reward": reward,
            "source": source,
            "license": license,
            "detail": {
                "gold": gold,
                "verdict": verdict_detail,
                "executorSteps": len(steps) - 1,
                "cost": meter.summary(),
                "plannerExecutorDisjoint": _disjoint(planner, executor),
                "appendOnly": planner.assert_append_only() and executor.assert_append_only(),
            },
        }

    # -- batch generation over a problem pack ------------------------------- #
    def generate_traces(self, problems: list[dict], *, reward_for: RewardFn | None = None) -> dict:
        """Run a rollout per problem ({prompt|goal, gold}); aggregate reward + cost."""
        traces: list[dict] = []
        agg = CostMeter(cache_rate=self.cache_rate)
        for prob in problems:
            goal = prob.get("prompt") or prob.get("goal") or ""
            tr = self.rollout(goal, gold=prob.get("gold"), reward_for=reward_for)
            traces.append(tr)
            c = tr["detail"]["cost"]
            agg.cached_input += c["cachedTotal"]
            agg.naive_input += c["naiveTotal"]
            agg.turns += c["turns"]
        rewards = [t["reward"] for t in traces if t["reward"] is not None]
        return {
            "traces": traces,
            "n": len(traces),
            "passRate": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "aggregateSavingsRatio": round((agg.naive_input or 0.0) / (agg.cached_input or 1.0), 3),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _render(session: Session) -> str:
    """Flatten the session into a single user payload (the OpenAI-compatible
    transports take one user string; the prefix stability is about *what* we send,
    not the wire format)."""
    return "\n".join(f"{m.role}: {m.content}" for m in session.messages)


def _summarize(session: Session) -> str:
    """Cheap deterministic compaction summary (offline). A live run would ask the
    model; here we keep the first goal + last reply, which is enough to preserve
    the rollout's thread without a network call."""
    first = session.messages[0].content[:200] if session.messages else ""
    last = session.messages[-1].content[:200] if session.messages else ""
    return f"goal≈{first} … latest≈{last}"


def _disjoint(a: Session, b: Session) -> bool:
    """Planner and executor must not share message objects (no prefix bleed)."""
    return not ({id(m) for m in a.messages} & {id(m) for m in b.messages})


def offline_invariants() -> "tuple[bool, dict]":
    """Assert the rollout-factory invariants (no torch, no GPU, no network).

    Proves the harness properties that make it a sound, cheap RLVR data pump:
    append-only prefixes, planner/executor session disjointness, a real cache
    saving on a multi-turn rollout, deterministic output, validated trajectories,
    and that the verifiable-reward seam reaches +1 on a correct answer.
    """
    from pipeline.rollout.cost import CostMeter
    from pipeline.rollout.session import Session
    from pretraining.vertical_data.schemas import validate_agent_trajectory
    from provenance_bench import physics_reward

    # 1) A correct rollout via a scripted executor -> reward 1.0, trace valid.
    correct = ScriptedClient(answer=r"\boxed{30 N}")
    f = RolloutFactory(client=correct)
    good = f.rollout("A 10 kg mass accelerates at 3 m/s^2; find the force.",
                     gold="30 N", reward_for=physics_reward.reward_for_problem)
    wrong_client = ScriptedClient(answer=r"\boxed{30 J}")  # right number, wrong unit
    fw = RolloutFactory(client=wrong_client)
    bad = fw.rollout("A 10 kg mass accelerates at 3 m/s^2; find the force.",
                     gold="30 N", reward_for=physics_reward.reward_for_problem)

    # 2) Deterministic: same scripted client, same input -> identical trace.
    f2 = RolloutFactory(client=ScriptedClient(answer=r"\boxed{30 N}"))
    good2 = f2.rollout("A 10 kg mass accelerates at 3 m/s^2; find the force.",
                       gold="30 N", reward_for=physics_reward.reward_for_problem)

    # 3) Cache savings on a deep single session (append-only beats naive).
    sess = Session(system="sys", context_window=100000)
    meter = CostMeter(cache_rate=0.1)
    sess.mark_sent()
    for i in range(8):
        cached = sess.cached_prefix_tokens
        user = f"turn {i}: " + ("context " * 40)
        sess.append("user", user)
        sess.append("assistant", "reply " * 20)
        sess.mark_sent()
        meter.record_turn(cached_prefix_tokens=cached,
                          fresh_input_tokens=count_tokens(user),
                          completion_tokens=count_tokens("reply " * 20))

    traj_check = validate_agent_trajectory({
        k: good[k] for k in ("goal", "steps", "outcome", "reward", "source", "license")
    })

    # 4) A multi-step executor tool loop deepens the session and uses calc as the
    #    arithmetic ground truth, ending in a correct \boxed answer.
    from pipeline.rollout.tools import DEFAULT_TOOLS

    tool_client = ScriptedClient(answers=["TOOL: calc(10*3)", r"\boxed{30 N}"])
    ft = RolloutFactory(client=tool_client)
    tool_run = ft.rollout("A 10 kg mass accelerates at 3 m/s^2; find the force.",
                          gold="30 N", reward_for=physics_reward.reward_for_problem,
                          tools=DEFAULT_TOOLS, max_executor_steps=4)

    checks = {
        "correctRewardOne": good["reward"] == 1.0,
        "wrongUnitRewardZero": bad["reward"] == 0.0,
        "appendOnly": good["detail"]["appendOnly"],
        "plannerExecutorDisjoint": good["detail"]["plannerExecutorDisjoint"],
        "deterministic": good["steps"] == good2["steps"] and good["reward"] == good2["reward"],
        "cacheSaves": meter.savings_ratio() > 1.5,
        "trajectoryValid": traj_check["ok"],
        "toolLoopRan": tool_run["detail"]["executorSteps"] >= 2,
        "toolLoopRewarded": tool_run["reward"] == 1.0,
        "toolLoopSaves": tool_run["detail"]["cost"]["savingsRatio"] > 1.0,
    }
    detail = {
        "checks": checks,
        "deepRolloutCost": meter.summary(),
        "toolRolloutCost": tool_run["detail"]["cost"],
        "goodReward": good["reward"],
        "badReward": bad["reward"],
        "sampleTrajectoryErrors": traj_check["errors"],
    }
    return all(checks.values()), detail
