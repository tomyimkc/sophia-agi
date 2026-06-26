# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the FlashAttention reproduction (numpy, no GPU).

The paper's core claim — tiled online-softmax attention is numerically identical
to O(N²) softmax attention while holding only one score tile — is verified here.
The fused Triton kernel is a GPU deployment artifact and is skipped in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from kernels import flash_attention as fa  # noqa: E402


def test_offline_invariants() -> None:
    ok, detail = fa.offline_invariants()
    assert ok, detail["checks"]


@pytest.mark.parametrize("shape", [(8, 8, 4), (33, 17, 8), (64, 64, 16), (1, 50, 8)])
def test_matches_naive_noncausal(shape) -> None:
    Nq, Nk, d = shape
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((Nq, d))
    K = rng.standard_normal((Nk, d))
    V = rng.standard_normal((Nk, d))
    ref = fa.naive_attention(Q, K, V)
    out = fa.flash_attention_numpy(Q, K, V, block_q=16, block_k=16)
    assert np.allclose(ref, out, atol=1e-9, rtol=1e-9)


def test_matches_naive_causal() -> None:
    rng = np.random.default_rng(2)
    Q = rng.standard_normal((40, 8))
    K = rng.standard_normal((40, 8))
    V = rng.standard_normal((40, 8))
    ref = fa.naive_attention(Q, K, V, causal=True)
    out = fa.flash_attention_numpy(Q, K, V, block_q=16, block_k=16, causal=True)
    assert np.allclose(ref, out, atol=1e-9, rtol=1e-9)


def test_numerically_stable_under_large_logits() -> None:
    rng = np.random.default_rng(3)
    big = rng.standard_normal((16, 8)) * 50.0
    ref = fa.naive_attention(big, big, big)
    out = fa.flash_attention_numpy(big, big, big)
    assert np.isfinite(out).all()
    assert np.allclose(ref, out, atol=1e-7, rtol=1e-7)


def test_block_size_invariance() -> None:
    rng = np.random.default_rng(4)
    Q = rng.standard_normal((40, 8))
    K = rng.standard_normal((40, 8))
    V = rng.standard_normal((40, 8))
    a = fa.flash_attention_numpy(Q, K, V, block_q=8, block_k=8, causal=True)
    b = fa.flash_attention_numpy(Q, K, V, block_q=40, block_k=40, causal=True)
    c = fa.flash_attention_numpy(Q, K, V, block_q=7, block_k=13, causal=True)
    assert np.allclose(a, b) and np.allclose(a, c)


def test_score_memory_is_sub_quadratic() -> None:
    rng = np.random.default_rng(5)
    N, d = 128, 16
    Q = rng.standard_normal((N, d))
    stats: dict = {}
    fa.flash_attention_numpy(Q, Q, Q, block_q=16, block_k=16, stats=stats)
    assert stats["max_score_tile"] <= 16 * 16
    assert stats["max_score_tile"] < stats["full_matrix"]  # < N^2


def test_triton_gated_off_in_ci() -> None:
    # No GPU in CI: the gated path must refuse rather than silently fall back.
    if not fa.triton_available():
        with pytest.raises(RuntimeError):
            fa.flash_attention_triton(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4)))
