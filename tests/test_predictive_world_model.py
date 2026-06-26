#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tabular_transition_model import PredictiveWorldModel, demo_world_model_report  # noqa: E402


def test_world_model_predicts_distribution_and_reward() -> None:
    wm = PredictiveWorldModel().fit([
        {"state": "s", "action": "a", "next_state": "x", "reward": 2},
        {"state": "s", "action": "a", "next_state": "x", "reward": 4},
        {"state": "s", "action": "b", "next_state": "y", "reward": -1},
        {"state": "s", "action": "b", "next_state": "y", "reward": -1},
    ])
    assert wm.predict_distribution("s", "a") == {"x": 1.0}
    assert wm.expected_reward("s", "a") == 3.0
    assert wm.choose_action("s", ["a", "b"])["chosen"]["action"] == "a"


def test_world_model_holds_ood() -> None:
    wm = PredictiveWorldModel()
    assert wm.choose_action("new", ["tool"])["verdict"] == "hold"


def test_world_model_demo_invariants() -> None:
    rep = demo_world_model_report()
    assert all(rep["invariants"].values())


def main() -> int:
    test_world_model_predicts_distribution_and_reward()
    test_world_model_holds_ood()
    test_world_model_demo_invariants()
    print("test_predictive_world_model: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
