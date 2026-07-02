#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic offline tests for Hadamard rotation (moe/rotation.py). No GPU."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe.rotation import (  # noqa: E402
    apply_input_rotation,
    block_absmax_stats,
    random_hadamard_matrix,
    rotate_activations,
    walsh_hadamard_matrix,
)


def test_hadamard_is_orthonormal():
    for n in (1, 2, 16, 2048):
        H = walsh_hadamard_matrix(n)
        I = H @ H.t()
        assert torch.allclose(I, torch.eye(n, dtype=I.dtype), atol=1e-9), f"n={n} not orthonormal"
    R = random_hadamard_matrix(256, seed=3)
    assert torch.allclose(R @ R.t(), torch.eye(256, dtype=R.dtype), atol=1e-9)


def test_rejects_non_power_of_two():
    for bad in (0, 3, 100):
        try:
            walsh_hadamard_matrix(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for n={bad}")


def test_paired_rotation_preserves_linear():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(64, 256, generator=g, dtype=torch.float64)
    x = torch.randn(256, 8, generator=g, dtype=torch.float64)
    R = random_hadamard_matrix(256, seed=1)
    Wr = apply_input_rotation(W, R)
    xr = rotate_activations(x, R)
    assert torch.allclose(Wr @ xr, W @ x, atol=1e-9), "paired (W Rᵀ)(R x) must equal W x"


def test_rotation_spreads_outlier_block_scale():
    # A weight with a few outlier COLUMNS: rotation mixes columns and spreads the outlier
    # energy across each row's 16-blocks, lowering the per-block absmax the FP4 scale keys on.
    g = torch.Generator().manual_seed(0)
    W = 0.01 * torch.randn(64, 256, generator=g, dtype=torch.float64)
    W[:, ::37] += 5.0  # sparse large-outlier columns
    R = random_hadamard_matrix(256, seed=2)
    Wr = apply_input_rotation(W, R)
    before = block_absmax_stats(W)["maxBlockAbsmax"]
    after = block_absmax_stats(Wr)["maxBlockAbsmax"]
    assert after < before, f"rotation must lower the max block absmax ({after} !< {before})"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
