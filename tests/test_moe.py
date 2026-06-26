# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for MoE routing + low-precision quant (numpy, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from moe import quant, router  # noqa: E402
from moe.router import MoERouter, load_balancing_loss, top_k_gating  # noqa: E402


# ---- router ----------------------------------------------------------------

def test_router_offline_invariants() -> None:
    ok, detail = router.offline_invariants()
    assert ok, detail["checks"]


def test_topk_gating_normalized() -> None:
    rng = np.random.default_rng(0)
    idx, combine, probs = top_k_gating(rng.standard_normal((16, 8)), k=2)
    assert np.allclose(probs.sum(1), 1.0)
    assert np.allclose(combine.sum(1), 1.0)
    assert idx.shape == (16, 2)


def test_aux_loss_floor_and_penalty() -> None:
    # balanced top-1 dispatch over 4 experts → aux == 1.0
    idx = (np.arange(64) % 4).reshape(-1, 1)
    probs = np.full((64, 4), 0.25)
    assert abs(load_balancing_loss(idx, probs, 4) - 1.0) < 1e-9
    # everyone to expert 0 → aux == E (worst case)
    idx_skew = np.zeros((64, 1), dtype=int)
    probs_skew = np.zeros((64, 4)); probs_skew[:, 0] = 1.0
    assert load_balancing_loss(idx_skew, probs_skew, 4) == pytest.approx(4.0)


def test_capacity_respected_and_drops_counted() -> None:
    r = MoERouter(4, k=2, capacity_factor=0.5, seed=1)
    logits = np.tile([10.0, 0.0, 0.0, 0.0], (40, 1))   # all want expert 0
    plan = r.route(logits)
    assert plan["dropped"] > 0
    assert all(c <= plan["capacity"] for c in plan["counts"])


def test_identity_experts_reconstruct_input() -> None:
    rng = np.random.default_rng(2)
    r = MoERouter(4, k=2, capacity_factor=4.0, seed=2)
    x = rng.standard_normal((30, 6))
    out, plan = r.forward(x, [lambda t: t] * 4)
    assert plan["dropped"] == 0
    assert np.allclose(out, x, atol=1e-9)


def test_router_requires_two_experts() -> None:
    with pytest.raises(ValueError):
        MoERouter(1)


# ---- quant -----------------------------------------------------------------

def test_quant_offline_invariants() -> None:
    ok, detail = quant.offline_invariants()
    assert ok, detail["checks"]


def test_int8_roundtrip_error_bounded() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 64))
    q, scale = quant.quantize_int8(W)
    dq = quant.dequantize_int8(q, scale)
    assert np.max(np.abs(dq - W)) <= scale / 2 + 1e-12
    assert q.min() >= -127 and q.max() <= 127


def test_per_channel_reduces_mean_error() -> None:
    rng = np.random.default_rng(1)
    W = rng.standard_normal((32, 16))
    W[:, 0] *= 100.0
    err_pt = np.mean(np.abs(quant.dequantize_int8(*quant.quantize_int8(W)) - W))
    err_pc = np.mean(np.abs(
        quant.dequantize_int8(*quant.quantize_int8(W, per_channel=True, axis=0)) - W))
    assert err_pc < err_pt / 5


def test_quantized_linear_close_to_fp() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 64))
    W = rng.standard_normal((64, 64))
    rel = np.linalg.norm(quant.quantized_linear(x, W) - x @ W) / np.linalg.norm(x @ W)
    assert rel < 0.02


def test_fp8_e4m3_relative_error_bound() -> None:
    rng = np.random.default_rng(4)
    vals = rng.uniform(0.1, 400.0, size=2000) * rng.choice([-1, 1], size=2000)
    fp8 = quant.fp8_e4m3_roundtrip(vals)
    assert np.max(np.abs(fp8 - vals) / np.abs(vals)) <= 2 ** -3 + 1e-9
    assert quant.fp8_e4m3_roundtrip([0.0])[0] == 0.0
    assert np.max(np.abs(quant.fp8_e4m3_roundtrip([1e9]))) == 448.0


def test_memory_reduction_factors() -> None:
    assert quant.int8_memory_reduction(32) == 4.0
    assert quant.int8_memory_reduction(16) == 2.0


# ---- NVFP4 (FP4 E2M1 + per-block micro-scale) ------------------------------

def test_nvfp4_snaps_to_e2m1_levels() -> None:
    rng = np.random.default_rng(0)
    snapped = quant._snap_e2m1(np.abs(rng.standard_normal(5000)) * 4.0)
    assert np.all(np.isin(snapped, quant._NVFP4_E2M1_LEVELS))


def test_nvfp4_zero_block_preserved() -> None:
    assert quant.nvfp4_roundtrip(np.zeros((1, quant.NVFP4_BLOCK)))[0, 0] == 0.0


def test_nvfp4_microscale_beats_global_fp4() -> None:
    # One giant block forces a huge global FP4 step that collapses the small blocks;
    # the per-block micro-scale recovers them. Judge on the non-max blocks.
    rng = np.random.default_rng(1)
    wide = rng.standard_normal((32, quant.NVFP4_BLOCK))
    wide[0] *= 100.0
    small = wide[1:]
    err_block = np.mean(np.abs(quant.nvfp4_roundtrip(wide)[1:] - small))
    g_scale = np.max(np.abs(wide)) / quant._NVFP4_E2M1_MAX
    err_global = np.mean(np.abs(
        np.sign(small) * quant._snap_e2m1(np.abs(small) / g_scale) * g_scale - small))
    assert err_block < err_global / 5


def test_nvfp4_weight_only_linear_close() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((8, 64))
    W = rng.standard_normal((64, 64))
    rel = np.linalg.norm(quant.quantized_linear_nvfp4(x, W) - x @ W) / np.linalg.norm(x @ W)
    assert rel < 0.15            # 4-bit: looser than INT8's 2%, still well-bounded


def test_nvfp4_memory_reduction_counts_microscale() -> None:
    # 4 bits + 8-bit scale per 16-element block = 4.5 effective bits.
    assert quant.nvfp4_memory_reduction(16) == pytest.approx(16.0 / 4.5)
    assert quant.nvfp4_memory_reduction(32) == pytest.approx(32.0 / 4.5)
    # Smaller blocks cost more scale overhead → less reduction.
    assert quant.nvfp4_memory_reduction(16, block_size=8) < quant.nvfp4_memory_reduction(16, block_size=32)
