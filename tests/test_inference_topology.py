# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the inference topology router (agent.inference_topology).
Dependency-free; exercises tier resolution + fail-closed behavior on a fixture config."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.inference_topology import (  # noqa: E402
    SCHEMA, TierSpec, _main, load_topology, resolve_tier, tier_spec, with_tier_env,
)

# A minimal v1 topology mirroring config/inference.local.spark.json (3 tiers).
FIXTURE = {
    "schema": SCHEMA,
    "hardware": {"machine": "DGX Spark (test)", "unified_memory_gb": 128},
    "tiers": {
        "orchestrator": {"role": "x", "engine": "vllm", "base_url": "http://localhost:8000/v1",
                         "model": "Qwen/Qwen2.5-14B-Instruct"},
        "tool_calls": {"role": "y", "engine": "sglang", "base_url": "http://localhost:30000/v1",
                       "model": "Qwen/Qwen2.5-7B-Instruct", "grammar": "json_schema"},
        "escalation": {"role": "z", "engine": "api", "provider": "anthropic",
                       "model": "claude-sonnet-4-6"},
    },
    "routing": {"default": "orchestrator", "structured_output": "tool_calls"},
}


def _write_fixture(tmp: Path) -> Path:
    p = tmp / "inference.local.json"
    p.write_text(json.dumps(FIXTURE), encoding="utf-8")
    return p


def test_resolve_local_tiers() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = _write_fixture(Path(tmp))
        topo = load_topology(p)
        orch = resolve_tier(topo, "orchestrator")
        assert isinstance(orch, TierSpec) and orch.provider == "vllm"
        assert orch.base_url == "http://localhost:8000/v1"
        assert orch.to_spec_str() == "vllm:Qwen/Qwen2.5-14B-Instruct"
        env = orch.to_env()
        assert env["SOPHIA_MODEL_PROVIDER"] == "vllm" and env["SOPHIA_MODEL_BASE_URL"].endswith(":8000/v1")
        tool = resolve_tier(topo, "tool_calls")
        assert tool.provider == "sglang" and tool.engine == "sglang"


def test_api_tier_keeps_provider() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = _write_fixture(Path(tmp))
        esc = tier_spec("escalation", path=p)
        assert esc is not None and esc.provider == "anthropic" and esc.model == "claude-sonnet-4-6"
        assert with_tier_env("escalation", path=p)["SOPHIA_MODEL_PROVIDER"] == "anthropic"


def test_fail_closed_when_absent() -> None:
    assert load_topology(Path("/nonexistent/inference.local.json")) is None
    assert resolve_tier(None, "orchestrator") is None
    assert tier_spec("orchestrator", path=Path("/nonexistent/x.json")) is None
    assert with_tier_env("orchestrator", path=Path("/nonexistent/x.json")) is None


def test_unknown_role_is_none() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = _write_fixture(Path(tmp))
        assert resolve_tier(load_topology(p), "nope") is None


def test_mlx_tier_refused() -> None:
    topo = {"schema": SCHEMA, "tiers": {"x": {"engine": "mlx", "model": "m"}}}
    try:
        resolve_tier(topo, "x")
    except ValueError as exc:
        assert "mlx" in str(exc).lower() and "apple-only" in str(exc).lower()
        return
    raise AssertionError("mlx tier should be refused fail-closed")


def test_api_tier_without_provider_refused() -> None:
    # fail-closed: an `api` engine tier with no `provider` must NOT default to openai
    topo = {"schema": SCHEMA, "tiers": {"x": {"engine": "api", "model": "m"}}}
    try:
        resolve_tier(topo, "x")
    except ValueError as exc:
        assert "api" in str(exc).lower() and "provider" in str(exc).lower()
        return
    raise AssertionError("api tier without provider should be refused fail-closed")


def test_unknown_engine_refused() -> None:
    topo = {"schema": SCHEMA, "tiers": {"x": {"engine": "weirdengine", "model": "m"}}}
    try:
        resolve_tier(topo, "x")
    except ValueError as exc:
        assert "unknown engine" in str(exc).lower()
        return
    raise AssertionError("unknown engine should be refused fail-closed")


def test_cli_reports_mlx_refusal_and_continues() -> None:
    import contextlib
    import io
    from unittest import mock

    topo = {
        "schema": SCHEMA,
        "hardware": {"machine": "Mac Studio test", "unified_memory_gb": 96},
        "tiers": {
            "orchestrator": {"engine": "mlx", "model": "mlx-model"},
            "tool_calls": {"engine": "llamacpp", "base_url": "http://localhost:8081/v1", "model": "gguf"},
            "escalation": {"engine": "api", "provider": "anthropic", "model": "claude-sonnet-4-6"},
        },
    }
    buf = io.StringIO()
    with mock.patch("agent.inference_topology.load_topology", lambda: topo), contextlib.redirect_stdout(buf):
        assert _main() == 0
    out = buf.getvalue()
    assert "orchestrator" in out and "refused" in out and "mlx" in out
    assert "tool_calls" in out and "llamacpp:gguf" in out
    assert "escalation" in out and "anthropic:claude-sonnet-4-6" in out


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_inference_topology: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
