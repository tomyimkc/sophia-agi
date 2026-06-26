# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Real RunPod LoRA backend (C1) — builds & governs jobs; never spends on its own.

This is the bridge from a search ``config`` to a REAL ``tools/runpod_train.py`` job and back
to a scalar objective. Its responsibilities are deliberately narrow and safe:

  * ``build_command(config)`` — the exact ``runpod_train.py`` argv. DRY-RUN unless
    ``execute=True`` is passed *and* the cost governor approves — and even then this module
    does not shell out for a paid run. Spending a real pod is done by the CI workflow
    (calibrate-runpod.yml) calling ``runpod_train.py --yes``, so a paid launch is always a
    human-triggered action, never a side effect of importing/looping here.
  * ``plan_trial(config)`` — cost projection (via ``CostGovernor``) + the dry-run command +
    a fail-closed budget guard. This is what the autonomous loop would call to decide.
  * ``score_result(eval_ladder_json)`` — parse the returned ladder into ``objective_for_min``.

Honest scope note: Step 1 (calibration) submits the DEFAULT pipeline to measure real
time/cost. Tuning LoRA rank / lr / mixture (the search space) is Step 2 and needs
``runpod_train.py`` to expose those passthrough args — flagged, not silently assumed.
"""
from __future__ import annotations

from typing import Any

from pretraining.autopilot.cost_governor import CostGovernor
from pretraining.autopilot.eval_ladder_objective import parse_objective

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


class RunPodLoRABackend:
    def __init__(self, governor: CostGovernor, *, branch: str,
                 model: str = DEFAULT_MODEL, train_data: str = "") -> None:
        self.gov = governor
        self.branch = branch
        self.model = model
        self.train_data = train_data

    def build_command(self, config: "dict[str, Any]", *, execute: bool = False) -> "list[str]":
        """The exact runpod_train.py argv. Dry-run unless execute (and even then, the caller
        — i.e. CI — is what actually runs it; this only constructs the command)."""
        cmd = ["python", "tools/runpod_train.py",
               "--branch", self.branch,
               "--model", config.get("model", self.model),
               "--epochs", str(config.get("epochs", 1)),
               "--seed", str(config.get("seed", 0))]
        train_data = config.get("train_data", self.train_data)
        if train_data:
            cmd += ["--train-data", train_data]
        # LoRA hyperparameter overrides (now threaded through runpod_train.py via
        # $SOPHIA_HPARAMS). Search-space key -> runpod_train.py flag.
        for key, flag in (("lr", "--lr"), ("lora_rank", "--lora-r"),
                          ("lora_alpha", "--lora-alpha"), ("lora_dropout", "--lora-dropout"),
                          ("neftune_alpha", "--neftune-alpha"), ("weight_decay", "--weight-decay")):
            if config.get(key) is not None:
                cmd += [flag, str(config[key])]
        if config.get("interruptible"):
            cmd += ["--interruptible"]   # cheaper spot pod
        cmd += ["--yes"] if execute else ["--dry-run"]
        return cmd

    def plan_trial(self, config: "dict[str, Any]") -> "dict[str, Any]":
        """Cost-governed plan for ONE trial. Fail-closed: if the projected spend would exceed
        the ceiling, ``affordable`` is False and the loop must stop. Never launches."""
        affordable = self.gov.can_afford(1)
        return {
            "launched": False,
            "affordable": affordable,
            "dry_run_command": " ".join(self.build_command(config, execute=False)),
            "launch_command": " ".join(self.build_command(config, execute=True))
                              + "   # CI-only, requires RUNPOD_API_KEY",
            "cost": self.gov.snapshot(),
            "guard": ("OK — within ceiling" if affordable else
                      "BLOCKED — projected spend exceeds ceiling; sweep should stop"),
        }

    def score_result(self, eval_ladder_json: dict, *, channel: str = "combined") -> "dict[str, Any]":
        """Parse a returned eval ladder into the objective (fail-closed)."""
        return parse_objective(eval_ladder_json, channel=channel)


__all__ = ["RunPodLoRABackend", "DEFAULT_MODEL"]
