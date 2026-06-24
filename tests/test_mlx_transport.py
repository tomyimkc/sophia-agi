#!/usr/bin/env python3
"""C3 MLX transport tests: the mlx provider resolves, reads the adapter env var, fails
closed (not crashes) when mlx_lm is absent, and is exempt from the airgap egress block."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.model import _call_mlx, _egress_blocked_for, default_client, resolve_config  # noqa: E402


def test_mlx_spec_resolves_with_adapter() -> None:
    os.environ["SOPHIA_MLX_ADAPTER"] = "training/mlx_adapters/sophia-v2"
    try:
        cfg = resolve_config("mlx:Qwen/Qwen2.5-3B-Instruct")
        assert cfg.kind == "mlx"
        assert cfg.model == "Qwen/Qwen2.5-3B-Instruct"
        assert cfg.adapter_path == "training/mlx_adapters/sophia-v2"
    finally:
        del os.environ["SOPHIA_MLX_ADAPTER"]


def test_non_mlx_spec_has_no_adapter() -> None:
    os.environ["SOPHIA_MLX_ADAPTER"] = "should-be-ignored"
    try:
        assert resolve_config("ollama:llama3.2").adapter_path is None
    finally:
        del os.environ["SOPHIA_MLX_ADAPTER"]


def test_mlx_fails_closed_without_mlx_lm() -> None:
    # mlx_lm is absent off-Mac/CI: the transport must return ok=False, not raise.
    cfg = resolve_config("mlx:Qwen/Qwen2.5-3B-Instruct")
    res = _call_mlx("sys", "user", cfg)
    assert res.ok is False
    assert res.provider == "mlx"
    assert "mlx_lm unavailable" in (res.error or "") or res.error  # graceful error string


def test_mlx_exempt_from_airgap() -> None:
    cfg = resolve_config("mlx:Qwen/Qwen2.5-3B-Instruct")
    prev = os.environ.get("SOPHIA_PROFILE")
    os.environ["SOPHIA_PROFILE"] = "airgap"
    try:
        # local on-device inference must never be blocked as "egress"
        assert _egress_blocked_for(cfg) is False
    finally:
        if prev is None:
            os.environ.pop("SOPHIA_PROFILE", None)
        else:
            os.environ["SOPHIA_PROFILE"] = prev


def test_default_client_builds_mlx() -> None:
    client = default_client("mlx:Qwen/Qwen2.5-3B-Instruct")
    assert client.primary.kind == "mlx"


def main() -> int:
    test_mlx_spec_resolves_with_adapter()
    test_non_mlx_spec_has_no_adapter()
    test_mlx_fails_closed_without_mlx_lm()
    test_mlx_exempt_from_airgap()
    test_default_client_builds_mlx()
    print("test_mlx_transport: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
