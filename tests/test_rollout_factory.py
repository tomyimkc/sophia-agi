# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the cache-stable rollout factory (pipeline/rollout).

Offline and deterministic — uses the ScriptedClient / mock backend, no network,
no GPU.
"""
from __future__ import annotations

from pipeline.rollout import (
    DEFAULT_TOOLS,
    CostMeter,
    RolloutFactory,
    ScriptedClient,
    Session,
    count_tokens,
    offline_invariants,
    safe_calc,
)
from pretraining.vertical_data.schemas import validate_agent_trajectory
from provenance_bench import physics_reward


# --------------------------------------------------------------------------- #
# Session: the append-only / cache-stability core
# --------------------------------------------------------------------------- #
def test_session_is_append_only() -> None:
    s = Session(system="sys")
    s.mark_sent()
    for i in range(5):
        s.append("user", f"u{i}")
        s.append("assistant", f"a{i}")
    assert s.assert_append_only()
    assert s.prefix_tokens() >= count_tokens("sys")


def test_compaction_resets_cold_prefix() -> None:
    s = Session(system="sys", context_window=40, compact_ratio=0.5)
    s.append("user", "go")
    s.append("assistant", "work " * 50)  # foldable bloat past 0.5*40 tokens
    assert s.needs_compaction()
    s.compact(summary="short summary")
    assert s.compactions == 1
    assert s.cached_prefix_tokens == 0  # cache cold after compaction
    assert not s.needs_compaction()     # assistant bloat folded; small user turn kept


# --------------------------------------------------------------------------- #
# CostMeter: caching is never more expensive and grows cheaper with depth
# --------------------------------------------------------------------------- #
def test_cache_savings_grow_with_depth() -> None:
    meter = CostMeter(cache_rate=0.1)
    prefix = 0
    for _ in range(10):
        meter.record_turn(cached_prefix_tokens=prefix, fresh_input_tokens=50,
                          completion_tokens=20)
        prefix += 70
    assert meter.savings_ratio() > 1.0
    assert meter.naive_total >= meter.cached_total


# --------------------------------------------------------------------------- #
# Factory: planner/executor split, verifiable reward, trace shape
# --------------------------------------------------------------------------- #
def test_rollout_correct_answer_rewarded() -> None:
    f = RolloutFactory(client=ScriptedClient(answer=r"\boxed{30 N}"))
    tr = f.rollout("10 kg at 3 m/s^2, find force", gold="30 N",
                   reward_for=physics_reward.reward_for_problem)
    assert tr["reward"] == 1.0
    assert tr["detail"]["plannerExecutorDisjoint"]
    assert tr["detail"]["appendOnly"]


def test_rollout_wrong_unit_not_rewarded() -> None:
    f = RolloutFactory(client=ScriptedClient(answer=r"\boxed{30 J}"))
    tr = f.rollout("10 kg at 3 m/s^2, find force", gold="30 N",
                   reward_for=physics_reward.reward_for_problem)
    assert tr["reward"] == 0.0


def test_trace_validates_as_agent_trajectory() -> None:
    f = RolloutFactory(client=ScriptedClient(answer=r"\boxed{30 N}"))
    tr = f.rollout("10 kg at 3 m/s^2, find force", gold="30 N",
                   reward_for=physics_reward.reward_for_problem)
    rec = {k: tr[k] for k in ("goal", "steps", "outcome", "reward", "source", "license")}
    assert validate_agent_trajectory(rec)["ok"]


def test_generate_traces_aggregates() -> None:
    f = RolloutFactory(client=ScriptedClient(answer=r"\boxed{30 N}"))
    problems = [
        {"prompt": "p1", "gold": "30 N"},
        {"prompt": "p2", "gold": "31 N"},  # scripted 30 N is >1% off -> reward 0
    ]
    out = f.generate_traces(problems, reward_for=physics_reward.reward_for_problem)
    assert out["n"] == 2
    assert out["passRate"] == 0.5
    assert out["aggregateSavingsRatio"] >= 1.0


def test_factory_runs_on_mock_backend() -> None:
    # No client passed -> real agent.model mock backend; harness must still run.
    f = RolloutFactory()
    tr = f.rollout("find the force", gold="30 N",
                   reward_for=physics_reward.reward_for_problem)
    assert tr["reward"] in (0.0, 1.0)
    assert len(tr["steps"]) == 2


# --------------------------------------------------------------------------- #
# Tools + multi-step executor loop (the cache win materializes with depth)
# --------------------------------------------------------------------------- #
def test_safe_calc_arithmetic() -> None:
    assert safe_calc("10*3") == "= 30"
    assert safe_calc("0.5 * 4 * 5**2") == "= 50"
    assert safe_calc("196 + 98") == "= 294"


def test_safe_calc_rejects_non_arithmetic() -> None:
    assert safe_calc("__import__('os').system('x')").startswith("error")
    assert safe_calc("open('f')").startswith("error")


def test_tool_loop_deepens_and_rewards() -> None:
    client = ScriptedClient(answers=["TOOL: calc(0.5*4*5**2)", r"\boxed{50 J}"])
    f = RolloutFactory(client=client)
    tr = f.rollout("4 kg at 5 m/s, find KE", gold="50 J",
                   reward_for=physics_reward.reward_for_problem,
                   tools=DEFAULT_TOOLS, max_executor_steps=4)
    assert tr["detail"]["executorSteps"] >= 2      # plan-less count: tool + execute
    assert tr["reward"] == 1.0
    # The calc observation is recorded as a step.
    assert any(s["action"].startswith("tool:calc") for s in tr["steps"])
    assert tr["detail"]["cost"]["savingsRatio"] > 1.0


# --------------------------------------------------------------------------- #
# Branching, best-of-N, stale-count loop, compaction archive (Reasonix ideas)
# --------------------------------------------------------------------------- #
def test_branch_shares_prefix() -> None:
    s = Session(system="sys")
    s.append("user", "plan")
    s.mark_sent()
    b = s.branch()
    assert b.cached_prefix_tokens == s.cached_prefix_tokens
    assert [m.content for m in b.messages] == [m.content for m in s.messages]
    # Diverging the branch must not touch the parent (no shared message objects).
    b.append("assistant", "branch-only")
    assert len(b.messages) == len(s.messages) + 1


def test_best_of_n_picks_passing_branch() -> None:
    client = ScriptedClient(answers=[r"\boxed{30 J}", r"\boxed{30 N}", r"\boxed{99 N}"])
    f = RolloutFactory(client=client)
    out = f.best_of_n("10 kg at 3 m/s^2, find force", gold="30 N",
                      reward_for=physics_reward.reward_for_problem, n=3)
    assert out["reward"] == 1.0
    assert out["detail"]["passes"] == 1
    assert out["detail"]["cost"]["savingsRatio"] > 1.5  # shared cached plan prefix


def test_generate_until_stops_when_solved() -> None:
    f = RolloutFactory(client=ScriptedClient(answer=r"\boxed{30 N}"))
    out = f.generate_until("find force", gold="30 N",
                           reward_for=physics_reward.reward_for_problem)
    assert out["detail"]["loop"]["solved"]
    assert out["detail"]["loop"]["attempts"] == 1


def test_generate_until_stops_when_stale() -> None:
    f = RolloutFactory(client=ScriptedClient(answer=r"\boxed{30 J}"))  # never right
    out = f.generate_until("find force", gold="30 N",
                           reward_for=physics_reward.reward_for_problem,
                           max_attempts=6, stale_cap=2)
    assert not out["detail"]["loop"]["solved"]
    assert out["detail"]["loop"]["stale"] >= 2
    assert out["detail"]["loop"]["attempts"] <= 6


def test_compaction_preserves_users_and_archives() -> None:
    s = Session(system="sys", context_window=40, compact_ratio=0.5)
    s.append("user", "the task")
    s.append("assistant", "work " * 50)
    archived: list = []
    s.compact(summary="digest", archive=lambda msgs: archived.extend(msgs))
    # User turn kept verbatim; assistant work archived, not kept.
    contents = [m.content for m in s.messages]
    assert "the task" in contents
    assert any(a.role == "assistant" for a in archived)
    assert all(m.role != "assistant" for m in s.messages)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]
