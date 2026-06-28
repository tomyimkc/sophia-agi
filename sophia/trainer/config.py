# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unified experiment config for the existing Sophia training workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11
    tomllib = None  # type: ignore[assignment]

SCHEMA = "sophia.experiment.v1"
CLAIM_BOUNDARY = (
    "AGI-candidate training workflow; not AGI proof. Promotion still requires "
    "verifier artifacts, external gates, and the no-overclaim standard."
)


def _as_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _as_args(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return tuple(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("boolean field must be true or false")
    return value


@dataclass(frozen=True)
class ModelConfig:
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    backend: str = "peft"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "ModelConfig":
        data = _as_mapping(payload, "model")
        return cls(
            name=str(data.get("name", cls.name)),
            backend=str(data.get("backend", cls.backend)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "backend": self.backend}


@dataclass(frozen=True)
class DataBuildConfig:
    enabled: bool = True
    script: str = "tools/prepare_lora_dataset.py"
    out_dir: str = "training/lora"
    max_tokens: int = 1024
    no_presplit: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DataBuildConfig":
        data = _as_mapping(payload, "data")
        return cls(
            enabled=_as_bool(data.get("enabled"), cls.enabled),
            script=str(data.get("script", cls.script)),
            out_dir=str(data.get("outDir", data.get("out_dir", cls.out_dir))),
            max_tokens=int(data.get("maxTokens", data.get("max_tokens", cls.max_tokens))),
            no_presplit=_as_bool(data.get("noPresplit", data.get("no_presplit")), cls.no_presplit),
            extra_args=_as_args(data.get("extraArgs", data.get("extra_args")), "data.extraArgs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "script": self.script,
            "outDir": self.out_dir,
            "maxTokens": self.max_tokens,
            "noPresplit": self.no_presplit,
            "extraArgs": list(self.extra_args),
        }


@dataclass(frozen=True)
class SFTConfig:
    enabled: bool = True
    script: str = "tools/train_lora.py"
    data: str = "training/sophia-math-code-curriculum/sft_all.jsonl"
    output: str = "training/lora/checkpoints/sophia-v1"
    epochs: int = 1
    seed: int = 0
    backend: str | None = None
    scaffold: bool = True
    guard: bool = True
    distill: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SFTConfig":
        data = _as_mapping(payload, "sft")
        return cls(
            enabled=_as_bool(data.get("enabled"), cls.enabled),
            script=str(data.get("script", cls.script)),
            data=str(data.get("data", cls.data)),
            output=str(data.get("output", cls.output)),
            epochs=int(data.get("epochs", cls.epochs)),
            seed=int(data.get("seed", cls.seed)),
            backend=data.get("backend"),
            scaffold=_as_bool(data.get("scaffold"), cls.scaffold),
            guard=_as_bool(data.get("guard"), cls.guard),
            distill=_as_bool(data.get("distill"), cls.distill),
            extra_args=_as_args(data.get("extraArgs", data.get("extra_args")), "sft.extraArgs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "script": self.script,
            "data": self.data,
            "output": self.output,
            "epochs": self.epochs,
            "seed": self.seed,
            "backend": self.backend,
            "scaffold": self.scaffold,
            "guard": self.guard,
            "distill": self.distill,
            "extraArgs": list(self.extra_args),
        }


@dataclass(frozen=True)
class DPOConfig:
    enabled: bool = False
    script: str = "tools/train_dpo.py"
    pairs: str = "training/local_sophia_7b/dpo_hard_negatives.jsonl"
    adapter: str = "training/lora/checkpoints/sophia-v1"
    output: str = "training/lora/checkpoints/sophia-dpo-v1"
    epochs: float = 1.0
    seed: int = 0
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DPOConfig":
        data = _as_mapping(payload, "dpo")
        return cls(
            enabled=_as_bool(data.get("enabled"), cls.enabled),
            script=str(data.get("script", cls.script)),
            pairs=str(data.get("pairs", cls.pairs)),
            adapter=str(data.get("adapter", cls.adapter)),
            output=str(data.get("output", cls.output)),
            epochs=float(data.get("epochs", cls.epochs)),
            seed=int(data.get("seed", cls.seed)),
            extra_args=_as_args(data.get("extraArgs", data.get("extra_args")), "dpo.extraArgs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "script": self.script,
            "pairs": self.pairs,
            "adapter": self.adapter,
            "output": self.output,
            "epochs": self.epochs,
            "seed": self.seed,
            "extraArgs": list(self.extra_args),
        }


@dataclass(frozen=True)
class RLVRConfig:
    enabled: bool = False
    script: str = "tools/run_rlvr.py"
    task: str = "provenance"
    reward: str = "verifier"
    output: str = "training/rlvr/checkpoints/sophia-rlvr-v1"
    report: str = "agi-proof/benchmark-results/rlvr.public-report.json"
    seed: int = 0
    # Real GRPO hyperparameters (map 1:1 to tools/run_rlvr.py args). Defaults mirror the
    # trainer's CLI defaults so a config that sets only `enabled:true` is still valid.
    # `model=None` means "resolve at plan time": mock under dry-run, else ExperimentConfig.model.
    model: str | None = None
    epochs: float = 1.0
    lr: float = 1e-5
    beta: float = 0.04
    num_generations: int = 8
    vllm: str = "colocate"
    quant: str = "4bit"
    max_prompt_len: int = 128
    max_completion_len: int = 128
    code_timeout: int = 15
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RLVRConfig":
        data = _as_mapping(payload, "rlvr")
        model = data.get("model")
        return cls(
            enabled=_as_bool(data.get("enabled"), cls.enabled),
            script=str(data.get("script", cls.script)),
            task=str(data.get("task", cls.task)),
            reward=str(data.get("reward", cls.reward)),
            output=str(data.get("output", cls.output)),
            report=str(data.get("report", cls.report)),
            seed=int(data.get("seed", cls.seed)),
            model=str(model) if model else None,
            epochs=float(data.get("epochs", cls.epochs)),
            lr=float(data.get("lr", cls.lr)),
            beta=float(data.get("beta", cls.beta)),
            num_generations=int(data.get("numGenerations", data.get("num_generations", cls.num_generations))),
            vllm=str(data.get("vllm", cls.vllm)),
            quant=str(data.get("quant", cls.quant)),
            max_prompt_len=int(data.get("maxPromptLen", data.get("max_prompt_len", cls.max_prompt_len))),
            max_completion_len=int(data.get("maxCompletionLen", data.get("max_completion_len", cls.max_completion_len))),
            code_timeout=int(data.get("codeTimeout", data.get("code_timeout", cls.code_timeout))),
            extra_args=_as_args(data.get("extraArgs", data.get("extra_args")), "rlvr.extraArgs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "script": self.script,
            "task": self.task,
            "reward": self.reward,
            "output": self.output,
            "report": self.report,
            "seed": self.seed,
            "model": self.model,
            "epochs": self.epochs,
            "lr": self.lr,
            "beta": self.beta,
            "numGenerations": self.num_generations,
            "vllm": self.vllm,
            "quant": self.quant,
            "maxPromptLen": self.max_prompt_len,
            "maxCompletionLen": self.max_completion_len,
            "codeTimeout": self.code_timeout,
            "extraArgs": list(self.extra_args),
        }


@dataclass(frozen=True)
class CalibrateConfig:
    """Low-RAM quantization calibration stage (disabled by default).

    Wires tools/run_calibration.py: build a deployment-distribution calibration set,
    prove it is disjoint from eval, and emit a datasheet to ship with a quantized
    artifact. Opt-in (enabled=False) so it never alters the default training plan.
    """

    enabled: bool = False
    script: str = "tools/run_calibration.py"
    sources: tuple[str, ...] = field(default_factory=tuple)
    out: str = "training/lora/calibration_datasheet.json"
    target_bits: float = 4.5
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CalibrateConfig":
        data = _as_mapping(payload, "calibrate")
        return cls(
            enabled=_as_bool(data.get("enabled"), cls.enabled),
            script=str(data.get("script", cls.script)),
            sources=_as_args(data.get("sources"), "calibrate.sources"),
            out=str(data.get("out", cls.out)),
            target_bits=float(data.get("targetBits", data.get("target_bits", cls.target_bits))),
            extra_args=_as_args(data.get("extraArgs", data.get("extra_args")), "calibrate.extraArgs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "script": self.script,
            "sources": list(self.sources),
            "out": self.out,
            "targetBits": self.target_bits,
            "extraArgs": list(self.extra_args),
        }


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool = True
    script: str = "tools/eval_ladder.py"
    backend: str = "hf"
    adapter: str = "training/lora/checkpoints/sophia-v1"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EvalConfig":
        data = _as_mapping(payload, "eval")
        return cls(
            enabled=_as_bool(data.get("enabled"), cls.enabled),
            script=str(data.get("script", cls.script)),
            backend=str(data.get("backend", cls.backend)),
            adapter=str(data.get("adapter", cls.adapter)),
            extra_args=_as_args(data.get("extraArgs", data.get("extra_args")), "eval.extraArgs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "script": self.script,
            "backend": self.backend,
            "adapter": self.adapter,
            "extraArgs": list(self.extra_args),
        }


@dataclass(frozen=True)
class PromotionConfig:
    enabled: bool = True
    script: str = "tools/promote_adapter.py"
    adapter_ladder: str = "training/local_sophia_v2/eval_ladder_adapter.json"
    baseline_ladder: str = "training/local_sophia_v2/eval_ladder_baseline.json"
    manifest: str = "training/local_sophia_v2/manifest.json"
    candidate_id: str = "local-sophia-v2-mlx"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PromotionConfig":
        data = _as_mapping(payload, "promotion")
        return cls(
            enabled=_as_bool(data.get("enabled"), cls.enabled),
            script=str(data.get("script", cls.script)),
            adapter_ladder=str(data.get("adapterLadder", data.get("adapter_ladder", cls.adapter_ladder))),
            baseline_ladder=str(data.get("baselineLadder", data.get("baseline_ladder", cls.baseline_ladder))),
            manifest=str(data.get("manifest", cls.manifest)),
            candidate_id=str(data.get("candidateId", data.get("candidate_id", cls.candidate_id))),
            extra_args=_as_args(
                data.get("extraArgs", data.get("extra_args")), "promotion.extraArgs"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "script": self.script,
            "adapterLadder": self.adapter_ladder,
            "baselineLadder": self.baseline_ladder,
            "manifest": self.manifest,
            "candidateId": self.candidate_id,
            "extraArgs": list(self.extra_args),
        }


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "sophia-local-candidate"
    schema: str = SCHEMA
    claim_boundary: str = CLAIM_BOUNDARY
    dry_run_default: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataBuildConfig = field(default_factory=DataBuildConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    rlvr: RLVRConfig = field(default_factory=RLVRConfig)
    calibrate: CalibrateConfig = field(default_factory=CalibrateConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(payload, dict):
            raise ValueError("experiment config must be an object")
        schema = str(payload.get("schema", SCHEMA))
        if schema != SCHEMA:
            raise ValueError(f"unsupported experiment schema: {schema}")
        return cls(
            name=str(payload.get("name", cls.name)),
            schema=schema,
            claim_boundary=str(
                payload.get("claimBoundary", payload.get("claim_boundary", CLAIM_BOUNDARY))
            ),
            dry_run_default=_as_bool(
                payload.get("dryRunDefault", payload.get("dry_run_default")),
                cls.dry_run_default,
            ),
            model=ModelConfig.from_dict(payload.get("model")),
            data=DataBuildConfig.from_dict(payload.get("data")),
            sft=SFTConfig.from_dict(payload.get("sft")),
            dpo=DPOConfig.from_dict(payload.get("dpo")),
            rlvr=RLVRConfig.from_dict(payload.get("rlvr")),
            calibrate=CalibrateConfig.from_dict(payload.get("calibrate")),
            eval=EvalConfig.from_dict(payload.get("eval")),
            promotion=PromotionConfig.from_dict(payload.get("promotion")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "name": self.name,
            "claimBoundary": self.claim_boundary,
            "dryRunDefault": self.dry_run_default,
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "sft": self.sft.to_dict(),
            "dpo": self.dpo.to_dict(),
            "rlvr": self.rlvr.to_dict(),
            "calibrate": self.calibrate.to_dict(),
            "eval": self.eval.to_dict(),
            "promotion": self.promotion.to_dict(),
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".toml":
        if tomllib is None:
            raise ValueError("TOML experiment configs require Python 3.11+")
        payload = tomllib.loads(text)
    else:
        payload = json.loads(text)
    return ExperimentConfig.from_dict(payload)
