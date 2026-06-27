# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Local/hybrid inference topology router — consumes config/inference.local.json.

This is the "future router wiring" referenced by ``config/inference.local.spark.json``.
A topology describes how to serve Sophia's model calls across local + cloud tiers
(the canonical example is an NVIDIA DGX Spark: a vLLM orchestrator, an SGLang
constrained-decoding tool-caller, and a cloud escalation tier). This module resolves a
*role* (``orchestrator`` / ``tool_calls`` / ``escalation`` / ...) to a concrete
``TierSpec`` that ``agent.model`` can call, and fail-closes when no topology is present
(returning ``None`` so the caller falls back to its default/cloud path).

It is dependency-free and config-only: it does NOT start servers (see
``scripts/spark_serve.sh``) and it never makes a network call. ``agent.model`` already
has ``vllm`` / ``sglang`` / ``openai``-compatible presets, so a tier just selects one.

Honest scope: routing is a policy lookup over a JSON config — NOT a load balancer, not
a scheduler, not a capability claim. ``mlx`` tiers are refused (Apple-only) so a Spark
config can't silently route to a backend that fails closed off-Mac.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "inference.local.json"
SCHEMA = "sophia.inference.local.v1"

# Engine -> Sophia provider preset (see agent.model._PRESETS). "api" tiers keep their
# own provider (anthropic/openai/...); "mlx" is refused (Apple-only).
_ENGINE_TO_PROVIDER = {
    "vllm": "vllm",
    "sglang": "sglang",
    "llamacpp": "llamacpp",
    "openai": "openai",
    "api": None,  # provider comes from the tier's own "provider" field
}


@dataclass(frozen=True)
class TierSpec:
    """A resolved inference tier: enough to call agent.model with, or to export as env."""

    role: str
    provider: str            # Sophia provider preset (vllm/sglang/anthropic/openai/...)
    model: str
    base_url: str | None = None
    engine: str = ""         # original engine from the config (vllm/sglang/api/...)
    notes: str = ""

    def to_spec_str(self) -> str:
        """``"provider:model"`` form accepted by agent.model.resolve_config / complete."""
        return f"{self.provider}:{self.model}"

    def to_env(self) -> dict[str, str]:
        """Env vars that route agent.model to this tier (SOPHIA_MODEL_PROVIDER/MODEL/BASE_URL)."""
        env: dict[str, str] = {"SOPHIA_MODEL_PROVIDER": self.provider, "SOPHIA_MODEL": self.model}
        if self.base_url:
            env["SOPHIA_MODEL_BASE_URL"] = self.base_url
        return env


def load_topology(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load the inference topology JSON, or return None if absent (fail-closed)."""
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{p}: expected schema {SCHEMA}, got {data.get('schema')!r}")
    return data


def resolve_tier(topology: dict[str, Any] | None, role: str) -> TierSpec | None:
    """Resolve ``role`` (e.g. orchestrator/tool_calls/escalation) to a TierSpec.

    Returns None if no topology or no such tier — callers fall back to their default
    path. Refuses ``mlx`` tiers (Apple-only) fail-closed.
    """
    if not topology:
        return None
    tiers = topology.get("tiers", {})
    tier = tiers.get(role)
    if not tier:
        return None
    engine = str(tier.get("engine", "")).lower()
    if engine == "mlx":
        raise ValueError(
            f"tier '{role}' uses engine 'mlx' (Apple-only); a Spark/aarch64 topology must use "
            "vllm/sglang/api. Refusing fail-closed."
        )
    provider = _ENGINE_TO_PROVIDER.get(engine)
    if provider is None:
        if engine == "api":
            # Fail-closed: an `api` tier MUST name its provider (anthropic/openai/...).
            # Defaulting silently to "openai" would route a self-hosted endpoint through the
            # wrong client preset (headers/auth/streaming shape) with no error.
            named = str(tier.get("provider", "")).strip()
            if not named:
                raise ValueError(
                    f"tier '{role}': engine 'api' requires a 'provider' field "
                    "(e.g. anthropic/openai); refusing fail-open."
                )
            provider = named
        else:
            raise ValueError(f"tier '{role}': unknown engine {engine!r}")
    return TierSpec(
        role=role,
        provider=provider,
        model=str(tier.get("model", "")),
        base_url=tier.get("base_url") or None,
        engine=engine,
        notes=str(tier.get("notes", "")),
    )


def tier_spec(role: str, *, path: str | Path | None = None) -> TierSpec | None:
    """Convenience: load the topology (once) and resolve one role. None if absent."""
    return resolve_tier(load_topology(path), role)


def with_tier_env(role: str, *, path: str | Path | None = None) -> dict[str, str] | None:
    """Return the env-var overlay for a role, or None if the tier is absent. Does NOT
    mutate os.environ — the caller decides whether to apply it (e.g. for one call)."""
    spec = tier_spec(role, path=path)
    return spec.to_env() if spec else None


def _main() -> int:
    topo = load_topology()
    if not topo:
        print(
            f"no topology at {DEFAULT_CONFIG} "
            "(copy config/inference.local.example.json or config/inference.local.spark.json -> config/inference.local.json)"
        )
        return 0
    hw = topo.get("hardware", {})
    print(f"topology: {hw.get('machine', '?')} ({hw.get('unified_memory_gb', '?')} GB unified)")
    for role in topo.get("tiers", {}):
        try:
            spec = resolve_tier(topo, role)
        except ValueError as exc:
            print(f"  {role:14s} !! refused: {exc}")
            continue
        if spec:
            print(f"  {role:14s} -> {spec.to_spec_str()}  base_url={spec.base_url or '-'}  [{spec.engine}]")
    routing = topo.get("routing", {})
    if routing:
        print(f"  routing: {routing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
