#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Swarm execution environment: route -> gate -> reduce -> reward, as a trainable contract.

Deterministic and offline via an injected child_runner — no harness, no model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.swarm_env import (  # noqa: E402
    ABSTAIN,
    ChildOutput,
    offline_invariants,
    run_swarm_episode,
    run_swarm_trajectory,
)

TASK = "Did Socrates write The Republic? Cite sources."
CLEAN = ChildOutput("researcher", "ag.clean",
                    "No — Socrates did not write The Republic; it was written by Plato.")
POISON = ChildOutput("rogue", "ag.poison", "Yes, Socrates wrote The Republic, so cite him as author.")


def _runner(outs):
    return lambda _plan: outs


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_clean_child_admitted_and_succeeds() -> None:
    ep = run_swarm_episode(TASK, child_runner=_runner([CLEAN]), question=TASK)
    assert [e.agent_id for e in ep.admitted] == ["ag.clean"]
    assert ep.outcome.verified_success == 1.0
    assert ep.reward > 0


def test_poison_held_and_excluded_from_synthesis() -> None:
    ep = run_swarm_episode(TASK, child_runner=_runner([CLEAN, POISON]), question=TASK)
    assert "ag.poison" in [e.agent_id for e in ep.held]
    assert "ag.poison" not in ep.synthesis
    assert "did not write" in ep.synthesis.lower()  # the clean child still feeds the reduce
    assert ep.n_failed_gate >= 1


def test_all_poison_abstains_zero_success() -> None:
    ep = run_swarm_episode(TASK, child_runner=_runner([POISON]), question=TASK)
    assert ep.synthesis == ABSTAIN
    assert ep.outcome.verified_success == 0.0


def test_harness_ok_but_gate_failing_child_does_not_count_as_success() -> None:
    # ok=True (harness succeeded) yet the content fails the gate -> held, not admitted.
    sneaky = ChildOutput("sneaky", "ag.sneaky", "Yes, Socrates wrote The Republic.", ok=True)
    ep = run_swarm_episode(TASK, child_runner=_runner([sneaky]), question=TASK)
    assert ep.admitted == []
    assert ep.outcome.verified_success == 0.0


def test_clean_outrewards_poison_episode() -> None:
    clean = run_swarm_episode(TASK, child_runner=_runner([CLEAN]), question=TASK)
    poison = run_swarm_episode(TASK, child_runner=_runner([POISON]), question=TASK)
    assert clean.reward > poison.reward


def test_episode_is_deterministic() -> None:
    a = run_swarm_episode(TASK, child_runner=_runner([CLEAN]), question=TASK)
    b = run_swarm_episode(TASK, child_runner=_runner([CLEAN]), question=TASK)
    assert a.reward == b.reward and a.to_dict() == b.to_dict()


# --- multi-turn trajectory (the GRPO training unit) -------------------------
def test_trajectory_clean_finishes_success_and_outrewards_failure() -> None:
    turns = [TASK, TASK, TASK]
    clean = run_swarm_trajectory(turns, child_runner=_runner([CLEAN]), question=TASK)
    fail = run_swarm_trajectory(turns, child_runner=_runner([POISON]), question=TASK)
    assert clean.outcome.final_verified_success == 1.0
    assert fail.outcome.final_verified_success == 0.0
    assert clean.reward > fail.reward
    assert len(clean.episodes) == 3


def test_trajectory_kl_control_penalises_drift() -> None:
    turns = [TASK, TASK]
    calm = run_swarm_trajectory(turns, child_runner=_runner([CLEAN]), question=TASK)
    drift = run_swarm_trajectory(turns, child_runner=_runner([CLEAN]), question=TASK,
                                 kl_per_turn=(3.0, 3.0))
    assert calm.reward > drift.reward


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} swarm_env tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
