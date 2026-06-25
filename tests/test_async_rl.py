# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the async-RL scaffolding (no torch, no GPU).

Proves the GRPO advantage math, the staleness-bounded replay buffer, and the
async-vs-sync scheduling trade-off deterministically. The real GPU GRPO step
stays gated in tools/run_rlvr.py.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import async_rl  # noqa: E402
from provenance_bench.async_rl import (  # noqa: E402
    ReplayBuffer,
    RolloutActor,
    Trajectory,
    grpo_advantages,
    scripted_generate,
    scripted_reward,
    simulate,
)


def test_offline_invariants() -> None:
    ok, detail = async_rl.offline_invariants()
    assert ok, detail["checks"]


def test_grpo_advantages_zero_mean_and_ordered() -> None:
    advs = grpo_advantages([2.0, 0.0, -2.0, 0.0])
    assert abs(sum(advs)) < 1e-9
    assert advs[0] > advs[1] > advs[2]


def test_grpo_degenerate_group_is_zero() -> None:
    assert grpo_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    assert grpo_advantages([]) == []


def test_buffer_drops_overstale() -> None:
    buf = ReplayBuffer(capacity=100, max_staleness=1)
    for v in range(5):
        buf.push(Trajectory(v, "p", "c", 1.0, 0.0, policy_version=v))
    batch = buf.sample(10, current_version=4)        # staleness 4-v
    assert all(4 - t.policy_version <= 1 for t in batch)
    assert buf.stats.dropped_stale == 3              # v=0,1,2


def test_buffer_capacity_bound() -> None:
    buf = ReplayBuffer(capacity=3, max_staleness=99)
    for v in range(10):
        buf.push(Trajectory(v, "p", "c", 1.0, 0.0, policy_version=v))
    assert len(buf) == 3
    assert buf.stats.dropped_overflow == 7


def test_actor_invokes_reward_seam() -> None:
    spy: dict = {}
    actor = RolloutActor(scripted_generate, scripted_reward, group_size=6)
    grp = actor.rollout(0, "c|skill=0.5", {}, 0, random.Random(0), spy=spy)
    assert spy["reward_calls"] == 6
    assert len(grp) == 6
    assert abs(sum(t.advantage for t in grp)) < 1e-6


def test_actor_requires_group_for_baseline() -> None:
    try:
        RolloutActor(scripted_generate, scripted_reward, group_size=1)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_async_beats_sync_throughput() -> None:
    a = simulate(mode="async", ticks=300, seed=3)
    s = simulate(mode="sync", ticks=300, seed=3)
    assert a.trained_trajectories > s.trained_trajectories
    assert a.trainer_idle_ticks < s.trainer_idle_ticks


def test_staleness_bound_respected() -> None:
    a = simulate(mode="async", ticks=300, max_staleness=2, seed=3)
    assert 0 < a.max_staleness_trained <= 2
    s = simulate(mode="sync", ticks=300, max_staleness=2, seed=3)
    assert s.max_staleness_trained == 0


def test_simulation_deterministic() -> None:
    assert simulate(mode="async", seed=9).as_dict() == simulate(mode="async", seed=9).as_dict()


def test_real_reward_seam_composes() -> None:
    """The async actor drops in over the repo's real provenance reward."""
    sys.path.insert(0, str(ROOT / "tools"))
    import run_async_rl  # noqa: E402

    out = run_async_rl._real_reward_smoke()
    assert out["advantages_zero_mean"]
    assert len(out["rewards"]) == 4
