#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Roofline harness math + the kernels orchestrator dry-run.

Pure CPU, offline, no GPU and no RunPod network. Verifies the roofline gate is honest
(<=100% of the ceiling, correct memory/compute-bound regime, over-100% guard) and that
tools/runpod_kernels.py builds a correct dry-run script without touching the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bench.roofline import (  # noqa: E402
    DEVICE_SPECS,
    analyze,
    attention_flops,
    gemm_bytes,
    gemm_flops,
    main as roofline_main,
    resolve_device,
)


def _a100():
    return DEVICE_SPECS["NVIDIA A100-SXM4-80GB"]


def test_gemm_accounting():
    assert gemm_flops(2, 3, 4) == 2 * 2 * 3 * 4
    # bytes: read A (m*k) + read B (k*n) + write C (m*n), 2 bytes each.
    assert gemm_bytes(2, 3, 4, 2) == 2 * (2 * 4 + 4 * 3 + 2 * 3)


def test_attention_causal_halves():
    full = attention_flops(1, 8, 1024, 64, causal=False)
    causal = attention_flops(1, 8, 1024, 64, causal=True)
    assert causal == full // 2


def test_compute_bound_pct_of_peak_exact():
    dev = _a100()
    m = n = k = 4096
    f, b = gemm_flops(m, n, k), gemm_bytes(m, n, k, 2)
    t = f / (dev.peak_flops("bf16") * 0.5)  # exactly 50% of compute peak
    r = analyze(flops=f, bytes_moved=b, times_s=[t, t, t], device=dev, dtype="bf16")
    assert r.regime == "compute-bound"
    assert r.pct_of_compute_peak == pytest.approx(0.5, abs=1e-3)
    assert r.pct_of_roofline <= 1.0 + 1e-6


def test_memory_bound_regime_and_bandwidth():
    dev = _a100()
    # k=1 -> tiny intensity -> memory bound.
    f, b = gemm_flops(4096, 4096, 1), gemm_bytes(4096, 4096, 1, 2)
    t = b / (dev.peak_bw() * 0.8)  # 80% of HBM
    r = analyze(flops=f, bytes_moved=b, times_s=[t, t, t], device=dev, dtype="bf16")
    assert r.regime == "memory-bound"
    assert r.pct_of_bandwidth_peak == pytest.approx(0.8, abs=1e-2)


def test_over_100pct_guard_fires():
    dev = _a100()
    f, b = gemm_flops(4096, 4096, 4096), gemm_bytes(4096, 4096, 4096, 2)
    t = f / (dev.peak_flops("bf16") * 2.0)  # impossible 200% of peak
    r = analyze(flops=f, bytes_moved=b, times_s=[t], device=dev, dtype="bf16")
    assert r.pct_of_roofline > 1.0
    assert any("roofline" in w for w in r.warnings)


def test_few_runs_warns():
    dev = _a100()
    f, b = gemm_flops(512, 512, 512), gemm_bytes(512, 512, 512, 2)
    t = f / (dev.peak_flops("bf16") * 0.3)
    r = analyze(flops=f, bytes_moved=b, times_s=[t, t], device=dev, dtype="bf16")
    assert any(">=3 runs" in w for w in r.warnings)


def test_resolve_device_fuzzy_and_unknown():
    assert resolve_device("h100 80gb hbm3").name == "NVIDIA H100 80GB HBM3"
    with pytest.raises(ValueError):
        resolve_device("Some Made Up GPU")


def test_fp4_peak_supported_only_on_blackwell():
    spark = resolve_device("DGX Spark GB10")
    # Spark has an FP4 tensor peak; it must exceed its BF16 peak (FP4 is the fast tier).
    assert spark.peak_flops("fp4") > spark.peak_flops("bf16")
    assert spark.peak_flops("nvfp4") == spark.peak_flops("fp4")
    # A pre-Blackwell card has no FP4 peak — requesting it raises, never guesses.
    with pytest.raises(ValueError):
        _a100().peak_flops("fp4")


def test_spark_decode_is_memory_bound_on_unified_lpddr():
    # Weight-streaming decode on the Spark: load a big FP4 weight (0.5 B/elem) for a
    # thin GEMM. The 273 GB/s unified wall dominates, so the regime must be memory-bound
    # and the headline % is bandwidth utilisation, not compute.
    spark = resolve_device("DGX Spark GB10")
    m, n, k = 1, 8192, 8192            # batch-1 token decode against an 8k x 8k weight
    f = gemm_flops(m, n, k)
    b = int(0.5 * n * k)               # 4-bit weights, the bytes that actually move
    t = b / (spark.peak_bw() * 0.7)    # hit 70% of the 273 GB/s wall
    r = analyze(flops=f, bytes_moved=b, times_s=[t, t, t], device=spark, dtype="fp4")
    assert r.regime == "memory-bound"
    assert r.pct_of_bandwidth_peak == pytest.approx(0.7, abs=1e-2)
    assert r.pct_of_roofline <= 1.0 + 1e-6


def test_invalid_inputs_rejected():
    dev = _a100()
    with pytest.raises(ValueError):
        analyze(flops=0, bytes_moved=1, times_s=[1.0], device=dev)
    with pytest.raises(ValueError):
        analyze(flops=1, bytes_moved=1, times_s=[0.0], device=dev)


def test_roofline_self_test_cli_exits_zero():
    assert roofline_main(["--self-test"]) == 0


def test_roofline_demo_cli_offline():
    # No --device, no torch -> falls back to a known device; must not raise.
    assert roofline_main(["--demo", "--device", "NVIDIA A100-SXM4-80GB", "--json"]) == 0


def test_kernels_orchestrator_dry_run_script():
    from tools.runpod_kernels import _remote_kernel_script, parse_args

    args = parse_args(["--dry-run", "--branch", "feature/x"])
    script = _remote_kernel_script(args)
    assert "feature/x" in script
    assert "roofline.py --self-test" in script
    assert "ncu" in script  # profiling hook present
    # dry-run main returns 0 and creates no pod.
    from tools.runpod_kernels import main as kmain

    assert kmain(["--dry-run"]) == 0
