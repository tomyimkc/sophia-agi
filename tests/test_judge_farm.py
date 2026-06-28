# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the local judge farm enablers (Spark-3):

- provenance_bench.aggregate._distinct_families counts a self-hosted vllm/sglang farm
  by MODEL VENDOR (like OpenRouter), so Qwen + Llama on two local ports = 2 families.
- agent.model.resolve_config honors a per-spec '@base_url' suffix so two local judges
  hit two distinct ports.
Dependency-free; no network."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clear_model_env(monkey: dict | None = None) -> None:
    for k in ("SOPHIA_MODEL_PROVIDER", "SOPHIA_MODEL_BASE_URL", "SOPHIA_MODEL"):
        os.environ.pop(k, None)


def test_local_judge_farm_is_two_families() -> None:
    from provenance_bench.aggregate import _distinct_families

    farm = [
        "vllm:Qwen/Qwen2.5-7B-Instruct@http://localhost:8000/v1",
        "vllm:meta-llama/Llama-3.3-8B-Instruct@http://localhost:8001/v1",
    ]
    assert _distinct_families(farm) == 2, "Qwen + Llama on local vLLM must be 2 families"


def test_same_vendor_local_judges_collapse() -> None:
    from provenance_bench.aggregate import _distinct_families

    assert _distinct_families(["vllm:Qwen/Qwen2.5-7B-Instruct", "vllm:Qwen/Qwen2.5-14B-Instruct"]) == 1


def test_aggregator_semantics_unchanged_for_cloud() -> None:
    from provenance_bench.aggregate import _distinct_families

    assert _distinct_families(["openrouter:anthropic/a", "openrouter:meta-llama/b"]) == 2
    assert _distinct_families(["anthropic:x", "deepseek:y"]) == 2
    assert _distinct_families(["anthropic:a", "anthropic:b"]) == 1


def test_spec_at_base_url_suffix_routes_to_port() -> None:
    from agent.model import resolve_config

    _clear_model_env()
    cfg = resolve_config("vllm:Qwen/Qwen2.5-7B-Instruct@http://localhost:8001/v1")
    assert cfg.model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.base_url == "http://localhost:8001/v1"
    assert cfg.label == "vllm"


def test_spec_without_suffix_uses_preset_default() -> None:
    from agent.model import resolve_config

    _clear_model_env()
    cfg = resolve_config("vllm:Qwen/Qwen2.5-7B-Instruct")
    assert cfg.base_url == "http://localhost:8000/v1"  # vllm preset default


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_judge_farm: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
