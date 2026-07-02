#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic offline tests for the GPTQ core (moe/gptq.py).

The load-bearing property: GPTQ minimizes OUTPUT error ``||(W-Wq)X||`` by compensating
with the inverse Hessian, so on correlated calibration activations it must not do worse
than round-to-nearest on that objective — and in practice does strictly better. No GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe.gptq import (  # noqa: E402
    gptq_quantize,
    output_mse,
    rtn_quantize,
    uniform_symmetric_quantizer,
)


def _correlated_case(seed: int = 0, rows: int = 64, cols: int = 128, nsamp: int = 512):
    """W [rows, cols] and X [cols, nsamp] whose feature-rows are correlated (non-diagonal H)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(rows, cols, generator=g)
    latent = torch.randn(rows, nsamp, generator=g)          # rows latent factors
    mix = torch.randn(cols, rows, generator=g)              # each of the cols features mixes them
    X = mix @ latent                                        # [cols, nsamp], correlated across cols
    H = X @ X.t()                                           # [cols, cols], non-diagonal
    return W, X, H


def test_gptq_not_worse_than_rtn_on_output_mse():
    W, X, H = _correlated_case()
    q = uniform_symmetric_quantizer(n_bits=4)
    rtn = rtn_quantize(W, q)
    gptq = gptq_quantize(W, H, q)
    mse_rtn = output_mse(W, rtn, X)
    mse_gptq = output_mse(W, gptq, X)
    # theoretical: GPTQ greedily minimizes this exact objective -> never worse (float slack)
    assert mse_gptq <= mse_rtn * (1 + 1e-6), f"gptq {mse_gptq} > rtn {mse_rtn}"
    # and on correlated activations it is meaningfully better
    assert mse_gptq < mse_rtn * 0.9, f"gptq {mse_gptq} not < 0.9*rtn {mse_rtn} (no compensation?)"


def test_gptq_preserves_shape_and_dtype():
    W, X, H = _correlated_case()
    W = W.to(torch.float16)
    gptq = gptq_quantize(W, H, uniform_symmetric_quantizer(4))
    assert gptq.shape == W.shape
    assert gptq.dtype == W.dtype


def test_gptq_rejects_mismatched_hessian():
    W, _, _ = _correlated_case()
    bad_H = torch.eye(W.shape[1] + 1)  # wrong size
    try:
        gptq_quantize(W, bad_H, uniform_symmetric_quantizer(4))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched H shape")


def test_gptq_zeroes_dead_columns():
    # A never-activated feature (all-zero across samples) -> zero Hessian diag -> weight zeroed.
    W, X, H = _correlated_case()
    H[3, :] = 0.0
    H[:, 3] = 0.0  # column 3 dead
    gptq = gptq_quantize(W, H, uniform_symmetric_quantizer(4))
    assert torch.all(gptq[:, 3] == 0), "dead column must be zeroed"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
