#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the installable Sophia package and dry-run trainer CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia.cli import main as sophia_main  # noqa: E402
from sophia.trainer import build_experiment_plan, load_experiment_config  # noqa: E402

SAMPLE_CONFIG = ROOT / "configs" / "sophia-experiment.sample.json"


def test_sample_config_compiles_all_stages_as_dry_run() -> None:
    cfg = load_experiment_config(SAMPLE_CONFIG)
    plan = build_experiment_plan(cfg)

    assert [spec.stage for spec in plan] == ["data", "sft", "dpo", "rlvr", "eval", "promotion"]
    assert all(spec.dry_run for spec in plan)
    assert all("--dry-run" in spec.command for spec in plan)
    assert cfg.claim_boundary.endswith("the no-overclaim standard.")

    rlvr = next(spec for spec in plan if spec.stage == "rlvr")
    assert rlvr.command[rlvr.command.index("--model") + 1] == "mock"
    assert rlvr.gpu_required_when_live


def test_live_plan_requires_explicit_dry_run_override() -> None:
    cfg = load_experiment_config(SAMPLE_CONFIG)
    plan = build_experiment_plan(cfg, dry_run=False)

    assert all("--dry-run" not in spec.command for spec in plan)
    rlvr = next(spec for spec in plan if spec.stage == "rlvr")
    assert rlvr.command[rlvr.command.index("--model") + 1] == cfg.model.name


def test_cli_plan_outputs_json_without_execution(capsys) -> None:
    rc = sophia_main(["experiment", "plan", "--config", str(SAMPLE_CONFIG), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "local-sophia-candidate-dry-run"
    assert [cmd["stage"] for cmd in payload["commands"]] == [
        "data",
        "sft",
        "dpo",
        "rlvr",
        "eval",
        "promotion",
    ]
    assert all(cmd["dryRun"] for cmd in payload["commands"])


def test_cli_train_limits_plan_to_training_stages(capsys) -> None:
    rc = sophia_main(["train", "--config", str(SAMPLE_CONFIG), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [cmd["stage"] for cmd in payload["commands"]] == ["data", "sft", "dpo", "rlvr"]


def main() -> int:
    test_sample_config_compiles_all_stages_as_dry_run()
    test_live_plan_requires_explicit_dry_run_override()
    print("test_sophia_package_cli: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
