# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the gather-on-read-set NVFP4 GEMM (Tier-2 bandwidth lever).

Pure CPU, no GPU. Tests the NumPy gather reference (full mask == dense; partial mask ==
dense-with-non-selected-rows-zeroed), the byte/FLOP accounting that proves traffic scales
with ρ, and the clean skip without CUDA. The Triton path runs on a real Spark/GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from kernels.src import gss_gather_gemm as gg  # noqa: E402
from kernels.src import nvfp4_gemm as ng  # noqa: E402


# ---- correctness oracle ----------------------------------------------------

def test_full_mask_equals_dense_nvfp4_gemm() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 256))
    W = rng.standard_normal((256, 64))
    mask = np.ones(gg.n_tiles(256, 64), dtype=bool)          # read every tile
    got = gg.gather_gemm_reference(x, W, mask, tile_size=64)
    dense = ng.nvfp4_gemm_reference(x, W)
    assert np.allclose(got, dense, atol=1e-9)                # gather full == dense, exactly


def test_partial_mask_equals_dense_with_zeroed_rows() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 256))
    W = rng.standard_normal((256, 48))
    ts = 64
    mask = np.array([True, False, True, False])              # 4 tiles, read tiles 0 and 2
    got = gg.gather_gemm_reference(x, W, mask, tile_size=ts)
    # equivalent: dense NVFP4 GEMM with the non-selected K-rows of W zeroed
    codes, scales, k0 = ng.quantize_nvfp4_weights(W)
    Wq = ng.dequantize_nvfp4_weights(codes, scales, k_orig=k0)
    Wz = Wq.copy()
    for t, keep in enumerate(mask):
        if not keep:
            Wz[t * ts:(t + 1) * ts, :] = 0.0
    assert np.allclose(got, x @ Wz, atol=1e-9)


def test_empty_mask_is_zero() -> None:
    x = np.ones((2, 128)); W = np.ones((128, 16))
    mask = np.zeros(gg.n_tiles(128, 64), dtype=bool)
    assert np.allclose(gg.gather_gemm_reference(x, W, mask, tile_size=64), 0.0)


def test_contraction_and_mask_shape_validated() -> None:
    x = np.ones((1, 100)); W = np.ones((128, 8))
    with pytest.raises(ValueError):
        gg.gather_gemm_reference(x, W, np.ones(2, bool), tile_size=64)   # K mismatch
    x = np.ones((1, 128))
    with pytest.raises(ValueError):
        gg.gather_gemm_reference(x, W, np.ones(5, bool), tile_size=64)   # wrong mask len


# ---- byte / FLOP accounting (the roofline denominator) ---------------------

def test_traffic_scales_with_rho() -> None:
    m, n, k, ts = 1, 8192, 8192, 256
    nt = gg.n_tiles(k, ts)
    dense = gg.gather_gemm_bytes(m, n, k, tile_size=ts, n_selected=nt)
    tenth = gg.gather_gemm_bytes(m, n, k, tile_size=ts, n_selected=nt // 10)
    ratio = tenth / dense
    assert 0.08 < ratio < 0.16          # ~ρ on the weight-dominated decode path
    # full selection == dense traffic exactly
    assert gg.gather_gemm_bytes(m, n, k, tile_size=ts, n_selected=nt) == dense


def test_flops_scale_with_selected_tiles() -> None:
    nt = gg.n_tiles(4096, 128)
    full = gg.gather_gemm_flops(1, 1024, 4096, tile_size=128, n_selected=nt)
    half = gg.gather_gemm_flops(1, 1024, 4096, tile_size=128, n_selected=nt // 2)
    assert abs(half / full - 0.5) < 0.02


def test_read_set_tiles_selects_fraction() -> None:
    mask = gg.read_set_tiles(0.1, 8192, 256, seed=3)
    nt = gg.n_tiles(8192, 256)
    assert mask.dtype == bool and mask.shape == (nt,)
    assert abs(mask.sum() / nt - 0.1) < 0.05
    with pytest.raises(ValueError):
        gg.read_set_tiles(0.0, 100, 10)


# ---- GPU path skips cleanly ------------------------------------------------

def test_gpu_path_skips_without_cuda() -> None:
    """Without torch+CUDA the runner returns None and exits clean (CI-green), like nvfp4."""
    ok, _ = gg._have_gpu_stack()
    if ok:
        pytest.skip("GPU stack present; skip-path not exercised")
    assert gg.run_gss_gather_gemm(verbose=False) is None
    assert gg.main([]) == 0
