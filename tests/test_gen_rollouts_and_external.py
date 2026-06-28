# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/gen_rollouts.py, the external physics scorer, and the physics
external-eval pack."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(mod: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gen_rollouts = _load("gen_rollouts", "tools/gen_rollouts.py")


# --------------------------------------------------------------------------- #
# gen_rollouts (verifier-gated SFT trace generation)
# --------------------------------------------------------------------------- #
def test_gen_rollouts_mock_runs_and_keeps_only_passing() -> None:
    rep = gen_rollouts.run("physics", model="mock", n=2, seed=0, max_problems=3)
    assert rep["problems"] == 3
    # mock never solves physics -> keepRate 0; every kept trace would be valid.
    assert 0.0 <= rep["keepRate"] <= 1.0
    assert rep["kept"] == len(rep["traces"])
    assert rep["invalid"] == 0


def test_gen_rollouts_kept_traces_validate() -> None:
    from pipeline.rollout import RolloutFactory, ScriptedClient
    from pretraining.vertical_data.schemas import validate_agent_trajectory
    from provenance_bench import physics_dataset, physics_reward

    rows = physics_dataset.build_physics_rl_dataset()["train_rows"][:1]
    gold = rows[0]["gold"]
    f = RolloutFactory(client=ScriptedClient(answers=[gold] * 4))
    bon = f.best_of_n(rows[0]["prompt"], gold=gold,
                      reward_for=lambda a, t=gold: physics_reward.reward_for_problem(a, t), n=4)
    assert bon["reward"] == 1.0
    rec = {"goal": bon["goal"], "steps": bon["steps"], "outcome": bon["outcome"],
           "reward": 1.0, "source": "test", "license": "Apache-2.0"}
    assert validate_agent_trajectory(rec)["ok"]


# --------------------------------------------------------------------------- #
# external physics scorer + pack
# --------------------------------------------------------------------------- #
def test_external_physics_scorer_dimension_aware() -> None:
    from agent import external_eval as ee
    assert ee.score_item_physics({"answer": "24 J"}, r"\boxed{24 J}")
    assert not ee.score_item_physics({"answer": "24 J"}, r"\boxed{24 N}")  # wrong dim
    assert not ee.score_item_physics({"answer": "24 J"}, r"\boxed{30 J}")  # wrong value


def test_external_physics_pack_self_consistent() -> None:
    from agent import external_eval as ee
    items = [json.loads(l) for l in (ROOT / "eval/external/physics-style-sample.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(items) >= 10
    # Feeding each gold back must score 100% (the pack's golds are self-consistent).
    res = ee.run_dataset(items, lambda it: it["answer"], scorer=ee.score_item_physics)
    assert res["accuracy"] == 1.0
    assert "dimensional" in res["oracle"]
