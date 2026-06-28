# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Thin command-plan wrappers around the existing training scripts."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sophia.trainer.config import ExperimentConfig

TRAINING_STAGES = ("data", "sft", "dpo", "rlvr")
CALIBRATION_STAGES = ("calibrate",)
EVAL_STAGES = ("eval",)
PROMOTION_STAGES = ("promotion",)
ALL_STAGES = (*TRAINING_STAGES, *CALIBRATION_STAGES, *EVAL_STAGES, *PROMOTION_STAGES)


@dataclass(frozen=True)
class CommandSpec:
    stage: str
    command: tuple[str, ...]
    description: str
    dry_run: bool
    gpu_required_when_live: bool = False

    def shell(self) -> str:
        return shlex.join(self.command)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "description": self.description,
            "dryRun": self.dry_run,
            "gpuRequiredWhenLive": self.gpu_required_when_live,
            "command": list(self.command),
            "shell": self.shell(),
        }


def _append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled and flag not in command:
        command.append(flag)


def _python(script: str) -> list[str]:
    return [sys.executable or "python3", script]


def _requested_stages(stages: Iterable[str] | None) -> tuple[str, ...]:
    if stages is None:
        return ALL_STAGES
    requested = tuple(stages)
    invalid = sorted(set(requested) - set(ALL_STAGES))
    if invalid:
        raise ValueError(f"unknown experiment stage(s): {', '.join(invalid)}")
    return requested


def build_experiment_plan(
    config: ExperimentConfig,
    *,
    stages: Iterable[str] | None = None,
    dry_run: bool | None = None,
) -> list[CommandSpec]:
    """Compile a Sophia experiment config into commands for the existing scripts.

    ``dry_run`` defaults to ``config.dry_run_default`` and keeps GPU-capable steps
    in wiring-check mode. Passing ``dry_run=False`` is an explicit live-run request.
    """
    use_dry_run = config.dry_run_default if dry_run is None else dry_run
    requested = set(_requested_stages(stages))
    plan: list[CommandSpec] = []

    if "data" in requested and config.data.enabled:
        command = [
            *_python(config.data.script),
            "--out-dir",
            config.data.out_dir,
            "--max-tokens",
            str(config.data.max_tokens),
            *config.data.extra_args,
        ]
        _append_flag(command, "--no-presplit", config.data.no_presplit)
        _append_flag(command, "--dry-run", use_dry_run)
        plan.append(
            CommandSpec(
                stage="data",
                command=tuple(command),
                description="Build LoRA train/holdout data with benchmark holdout guards.",
                dry_run=use_dry_run,
            )
        )

    if "sft" in requested and config.sft.enabled:
        command = [
            *_python(config.sft.script),
            "--model",
            config.model.name,
            "--data",
            config.sft.data,
            "--output",
            config.sft.output,
            "--backend",
            config.sft.backend or config.model.backend,
            "--epochs",
            str(config.sft.epochs),
            "--seed",
            str(config.sft.seed),
            *config.sft.extra_args,
        ]
        _append_flag(command, "--scaffold", config.sft.scaffold)
        _append_flag(command, "--guard", config.sft.guard)
        _append_flag(command, "--distill", config.sft.distill)
        _append_flag(command, "--dry-run", use_dry_run)
        plan.append(
            CommandSpec(
                stage="sft",
                command=tuple(command),
                description="Run supervised LoRA fine-tuning through tools/train_lora.py.",
                dry_run=use_dry_run,
                gpu_required_when_live=(config.sft.backend or config.model.backend) != "mlx",
            )
        )

    if "dpo" in requested and config.dpo.enabled:
        command = [
            *_python(config.dpo.script),
            "--model",
            config.model.name,
            "--pairs",
            config.dpo.pairs,
            "--adapter",
            config.dpo.adapter,
            "--output",
            config.dpo.output,
            "--epochs",
            str(config.dpo.epochs),
            "--seed",
            str(config.dpo.seed),
            *config.dpo.extra_args,
        ]
        _append_flag(command, "--dry-run", use_dry_run)
        plan.append(
            CommandSpec(
                stage="dpo",
                command=tuple(command),
                description="Run DPO hard-negative preference tuning via tools/train_dpo.py.",
                dry_run=use_dry_run,
                gpu_required_when_live=True,
            )
        )

    if "rlvr" in requested and config.rlvr.enabled:
        # Model resolution: an explicit per-stage model wins; else mock under dry-run
        # (offline reward-wiring check); else the experiment model. dry_run_default stays
        # the config default, so a careless `sophia experiment run` never spends GPU.
        if config.rlvr.model:
            model = config.rlvr.model
        elif use_dry_run:
            model = "mock"
        else:
            model = config.model.name
        command = [
            *_python(config.rlvr.script),
            "--model",
            model,
            "--task",
            config.rlvr.task,
            "--reward",
            config.rlvr.reward,
            "--output",
            config.rlvr.output,
            "--out",
            config.rlvr.report,
            "--seed",
            str(config.rlvr.seed),
            "--epochs",
            str(config.rlvr.epochs),
            "--lr",
            str(config.rlvr.lr),
            "--beta",
            str(config.rlvr.beta),
            "--num-generations",
            str(config.rlvr.num_generations),
            "--vllm",
            config.rlvr.vllm,
            "--quant",
            config.rlvr.quant,
            "--max-prompt-len",
            str(config.rlvr.max_prompt_len),
            "--max-completion-len",
            str(config.rlvr.max_completion_len),
            *config.rlvr.extra_args,
        ]
        if config.rlvr.task == "code":
            command += ["--code-timeout", str(config.rlvr.code_timeout)]
        _append_flag(command, "--dry-run", use_dry_run)
        plan.append(
            CommandSpec(
                stage="rlvr",
                command=tuple(command),
                description="Run verifier-as-reward/RLVR wiring or live GRPO via tools/run_rlvr.py.",
                dry_run=use_dry_run,
                gpu_required_when_live=True,
            )
        )

    if "calibrate" in requested and config.calibrate.enabled:
        command = [
            *_python(config.calibrate.script),
            "--out",
            config.calibrate.out,
            "--target-bits",
            str(config.calibrate.target_bits),
        ]
        if config.calibrate.sources:
            command += ["--sources", *config.calibrate.sources]
        command += list(config.calibrate.extra_args)
        _append_flag(command, "--dry-run", use_dry_run)
        plan.append(
            CommandSpec(
                stage="calibrate",
                command=tuple(command),
                description="Build a decontaminated low-RAM quantization calibration datasheet "
                            "via tools/run_calibration.py.",
                dry_run=use_dry_run,
            )
        )

    if "eval" in requested and config.eval.enabled:
        command = [
            *_python(config.eval.script),
            "--model",
            config.model.name,
            "--adapter",
            config.eval.adapter,
            "--backend",
            config.eval.backend,
            *config.eval.extra_args,
        ]
        _append_flag(command, "--dry-run", use_dry_run)
        plan.append(
            CommandSpec(
                stage="eval",
                command=tuple(command),
                description="Run the local-model eval ladder through tools/eval_ladder.py.",
                dry_run=use_dry_run,
            )
        )

    if "promotion" in requested and config.promotion.enabled:
        command = [
            *_python(config.promotion.script),
            "--adapter-ladder",
            config.promotion.adapter_ladder,
            "--baseline-ladder",
            config.promotion.baseline_ladder,
            "--manifest",
            config.promotion.manifest,
            "--candidate-id",
            config.promotion.candidate_id,
            *config.promotion.extra_args,
        ]
        _append_flag(command, "--dry-run", use_dry_run)
        plan.append(
            CommandSpec(
                stage="promotion",
                command=tuple(command),
                description="Evaluate adapter promotion through the no-overclaim gate.",
                dry_run=use_dry_run,
            )
        )

    return plan


def execute_plan(plan: Iterable[CommandSpec], *, cwd: str | Path) -> int:
    """Execute a prebuilt command plan in order, stopping on the first failure."""
    root = Path(cwd)
    for spec in plan:
        print(f"[{spec.stage}] {spec.shell()}", flush=True)
        proc = subprocess.run(spec.command, cwd=root, text=True)
        if proc.returncode != 0:
            return proc.returncode
    return 0
