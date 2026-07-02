#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the NVFP4 group-16 GPTQ core (the served-reproducible path). No GPU.

The load-bearing properties: (1) ``nvfp4_group_quantize`` is BIT-IDENTICAL to the flattened
``_torch_nvfp4`` grid the server uses (grid-identity foundation); (2) grouped GPTQ minimizes
OUTPUT error, so on correlated activations it beats round-to-nearest on the SAME grid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from moe.gptq import (  # noqa: E402
    gptq_quantize_grouped,
    nvfp4_group_quantize,
    output_mse,
)


def _torch_nvfp4_ref(w):
    """Reference copy of training.qat._torch_nvfp4 (flatten -> block-16 -> amax/6 -> E2M1)."""
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=w.dtype)
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=w.dtype)
    flat = w.reshape(-1)
    pad = (-flat.numel()) % 16
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, 16)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / 6.0
    idx = torch.bucketize((blocks / scale).abs(), bounds)
    dq = torch.sign(blocks) * levels[idx] * scale
    return dq.reshape(-1)[: w.numel()].reshape(w.shape)


def test_nvfp4_group_bit_identical_to_served_flatten_grid():
    # GRID IDENTITY: for in-dim a multiple of 16, group-16-along-input == the flattened
    # _torch_nvfp4 grid the server serves, bit-for-bit. This is what makes GPTQ served-reproducible.
    g = torch.Generator().manual_seed(0)
    W = torch.randn(32, 64, generator=g)
    mine = nvfp4_group_quantize(W, group_size=16)
    ref = _torch_nvfp4_ref(W)
    assert torch.equal(mine, ref), f"max|Δ|={(mine-ref).abs().max()}"


def test_nvfp4_group_values_on_grid():
    g = torch.Generator().manual_seed(1)
    W = torch.randn(16, 32, generator=g)
    q = nvfp4_group_quantize(W)
    # every value is sign * level * (per-group scale); check |q/scale| lands on a level
    Wf = W.reshape(16, 2, 16)
    scale = Wf.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / 6.0
    ratio = (q.reshape(16, 2, 16) / scale).abs()
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    on_grid = (ratio.unsqueeze(-1) - levels).abs().min(dim=-1).values
    assert float(on_grid.max()) < 1e-4


def test_grouped_gptq_beats_rtn_on_same_grid():
    g = torch.Generator().manual_seed(0)
    rows, cols, nsamp = 32, 64, 256
    W = torch.randn(rows, cols, generator=g)
    latent = torch.randn(rows, nsamp, generator=g)
    mix = torch.randn(cols, rows, generator=g)
    X = mix @ latent                       # [cols, nsamp], correlated across cols
    H = X @ X.t()
    rtn = nvfp4_group_quantize(W, group_size=16)
    gptq = gptq_quantize_grouped(W, H, group_size=16)
    mse_rtn = output_mse(W, rtn, X)
    mse_gptq = output_mse(W, gptq, X)
    assert mse_gptq <= mse_rtn * (1 + 1e-6), f"gptq {mse_gptq} > rtn {mse_rtn}"
    assert mse_gptq < mse_rtn * 0.9, f"gptq {mse_gptq} not < 0.9*rtn {mse_rtn} (no compensation?)"


def test_grouped_rejects_non_multiple_indim():
    W = torch.randn(8, 20)  # 20 not a multiple of 16
    H = torch.eye(20)
    for fn in (lambda: nvfp4_group_quantize(W), lambda: gptq_quantize_grouped(W, H)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for non-multiple-of-16 in-dim")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
