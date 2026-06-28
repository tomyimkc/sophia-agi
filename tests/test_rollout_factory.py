# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the cache-stable rollout factory (pipeline/rollout).

Offline and deterministic — uses the ScriptedClient / mock backend, no network,
no GPU.
"""
from __future__ import annotations

from pipeline.rollout import (
    CostMeter,
    RolloutFactory,
    ScriptedClient,
    Session,
    count_tokens,
    offline_invariants,
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
    s.append("user", "x " * 50)  # blow past 0.5*40 tokens
    assert s.needs_compaction()
    s.compact(summary="short summary")
    assert s.compactions == 1
    assert s.cached_prefix_tokens == 0  # cache cold after compaction
    assert not s.needs_compaction()


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


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]
