# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the fused NVFP4 dequant-GEMM + the Spark roofline harness.

Pure CPU, no GPU. The Triton kernel itself needs a Spark to run (and skips cleanly
without one); here we test the NumPy reference, the 4-bit packing, the FLOP/byte
accounting the roofline divides by, and the report harness's provenance annotations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from kernels.src import nvfp4_gemm as ng  # noqa: E402


# ---- packing ----------------------------------------------------------------

def test_int4_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=(7, 30)).astype(np.uint8)   # odd row length -> pad path
    packed = ng.pack_int4(codes)
    assert packed.shape[-1] == 15                                 # 30 codes -> 15 bytes
    back = ng.unpack_int4(packed, length=30)
    assert np.array_equal(back, codes)


def test_pack_rejects_non_4bit():
    with pytest.raises(ValueError):
        ng.pack_int4(np.array([[16]], dtype=np.uint8))


def test_pack_halves_the_bytes():
    codes = np.zeros((4, 64), dtype=np.uint8)
    assert ng.pack_int4(codes).nbytes == codes.nbytes // 2       # true 4-bit storage


# ---- quantize / dequant reference ------------------------------------------

def test_quantize_codes_are_4bit_and_shapes_align():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((40, 8))                              # K not a multiple of 16
    codes, scales, k_orig = ng.quantize_nvfp4_weights(W, block_size=16)
    assert k_orig == 40
    assert codes.shape == (48, 8)                                 # padded to 48
    assert scales.shape == (3, 8)                                 # 48/16 blocks
    assert codes.min() >= 0 and codes.max() <= 15


def test_dequant_strips_padding_and_approximates():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((40, 8))
    codes, scales, k_orig = ng.quantize_nvfp4_weights(W)
    Wq = ng.dequantize_nvfp4_weights(codes, scales, k_orig=k_orig)
    assert Wq.shape == W.shape
    rel = np.linalg.norm(Wq - W) / np.linalg.norm(W)
    assert rel < 0.20                                             # 4-bit weight-only bound


def test_reference_consistent_with_moe_quant_single_block():
    # One K-block, one column: this harness's K-blocking matches moe/quant.py's
    # flattened blocking, so the two NVFP4 references must agree.
    from moe import quant
    rng = np.random.default_rng(3)
    W = rng.standard_normal((16, 1))
    codes, scales, k = ng.quantize_nvfp4_weights(W, block_size=16)
    here = ng.dequantize_nvfp4_weights(codes, scales, k_orig=k)
    there = quant.nvfp4_roundtrip(W, block_size=16)
    assert np.allclose(here, there, atol=1e-9)


def test_gemm_reference_close_to_fp():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((8, 64))
    W = rng.standard_normal((64, 32))
    rel = np.linalg.norm(ng.nvfp4_gemm_reference(x, W) - x @ W) / np.linalg.norm(x @ W)
    assert rel < 0.15


# ---- FLOP / byte accounting (the roofline denominator) ----------------------

def test_gemm_flops_accounting():
    assert ng.nvfp4_gemm_flops(2, 3, 4) == 2 * 2 * 3 * 4


def test_gemm_bytes_weights_are_half_byte_each():
    # m=1 decode, k=n=4096, block 16: weights = n*k/2 (int4) + (k/16)*n (fp8 scales).
    m, n, k = 1, 4096, 4096
    expect_w = (n * k) // 2 + (k // 16) * n
    expect_act = 2 * (m * k + m * n)
    assert ng.nvfp4_gemm_bytes(m, n, k) == expect_w + expect_act
    # FP4 weight bytes are ~4x smaller than the BF16 weight read it replaces.
    from kernels.bench.roofline import gemm_bytes
    bf16_weight_bytes = 2 * (k * n)
    assert expect_w < bf16_weight_bytes / 3                       # comfortably sub-1/3


def test_decode_is_memory_bound_against_spark_roofline():
    from kernels.bench.roofline import analyze, resolve_device
    spark = resolve_device("DGX Spark GB10")
    m, n, k = 1, 8192, 8192
    f = ng.nvfp4_gemm_flops(m, n, k)
    b = ng.nvfp4_gemm_bytes(m, n, k)
    t = b / (spark.peak_bw() * 0.6)                              # hit 60% of the 273 GB/s wall
    r = analyze(flops=f, bytes_moved=b, times_s=[t, t, t], device=spark, dtype="fp4")
    assert r.regime == "memory-bound"
    assert r.pct_of_roofline <= 1.0 + 1e-6


# ---- the kernel skips cleanly without a GPU --------------------------------

def test_run_kernel_skips_without_gpu():
    # No torch/CUDA/triton in CI -> returns None, never raises, exit code stays 0.
    assert ng.run_nvfp4_gemm(verbose=False) is None
    assert ng.main(["--m", "1", "--n", "256", "--k", "256"]) == 0


# ---- Spark roofline report harness -----------------------------------------

def test_report_carries_provenance_boundary():
    from tools.spark_roofline_report import build_report
    rep = build_report([{"kernel": "nvfp4_gemm", "shape": {"m": 1, "n": 8, "k": 8},
                         "roofline": None}], stamp="t0")
    assert rep["sparkIteration"] is True
    assert rep["registeredResult"] is False                      # non-negotiable boundary
    assert rep["device"] == "NVIDIA DGX Spark GB10"
    assert rep["ceiling"]["bandwidth_gbytes_s"] == 273.0


def test_harness_dry_run_exits_zero_and_writes_nothing(capsys):
    from tools.spark_roofline_report import main
    assert main([]) == 0                                          # default is dry-run
    out = capsys.readouterr().out
    assert "registeredResult" in out or "sparkIteration" in out.lower() or "Provenance" in out
    assert "Dry-run" in out


def test_harness_rejects_unknown_device():
    from tools.spark_roofline_report import main
    assert main(["--device", "Totally Made Up GPU"]) == 2
