#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Phase-4 Spark-MoE serve helper (pure-plan, offline)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import serve_spark_moe as ssm  # noqa: E402


def test_plan_fp8_command_and_env() -> None:
    p = ssm.plan(model="Qwen/Qwen3-Next-80B-A3B", quant="fp8",
                 max_model_len=32768, port=8000, gpu_mem_util=0.9)
    cmd = " ".join(p["serveCmd"])
    assert "vllm serve Qwen/Qwen3-Next-80B-A3B" in cmd
    assert "--quantization fp8" in cmd
    assert "--port 8000" in cmd
    assert "--max-model-len 32768" in cmd
    assert "nvfp4" not in cmd.lower()  # never emit the broken quant
    assert p["sophiaEnv"]["SOPHIA_MODEL_PROVIDER"] == "vllm"
    assert p["sophiaEnv"]["SOPHIA_MODEL_BASE_URL"] == "http://localhost:8000/v1"
    assert p["forcedFromNvfp4"] is False


def test_nvfp4_is_forced_to_fp8() -> None:
    p = ssm.plan(model="X/Y", quant="nvfp4", max_model_len=8192, port=9000, gpu_mem_util=0.9)
    assert p["quant"] == "fp8"
    assert p["forcedFromNvfp4"] is True
    assert "nvfp4" not in " ".join(p["serveCmd"]).lower()


def test_none_quant_omits_flag() -> None:
    p = ssm.plan(model="X/Y", quant="None", max_model_len=4096, port=8000, gpu_mem_util=0.9)
    assert "--quantization" not in " ".join(p["serveCmd"])
