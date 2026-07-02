# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hadamard weight rotation for NVFP4 — spread outlier energy before the FP4 snap (旋转抗离群).

NVFP4's per-block-16 absmax micro-scale (``training.qat._torch_nvfp4``) is dominated by the
largest weight in each block: one outlier forces a coarse scale for the whole block. A random
Walsh-Hadamard rotation ``R`` (orthonormal) Gaussianizes the weight rows and spreads outlier
energy across the block, shrinking the per-block dynamic range the 8-level E2M1 grid must cover
(QuaRot / SpinQuant / MicroRot; Ashkboos et al. 2024, arXiv:2404.00456).

**Computational invariance (the load-bearing caveat).** Rotating a weight is only *served-
reproducible* under PAIRED ABSORPTION: for ``y = W x``, replacing ``W -> W Rᵀ`` requires the
input to arrive pre-rotated ``x -> R x`` (absorbed into the previous layer's output projection,
which commutes with RMSNorm for an orthonormal ``R``). A rotation applied ONLY inside the cert's
``quantize_served_params`` produces a number a stock ``--quantization nvfp4`` server CANNOT
reproduce. So this module ships the ROTATION PRIMITIVE + a paired (weight, activation) rotator
for an HONEST upper-bound probe; the full residual-stream absorption into the served graph is the
follow-on before any certifiable number. ``canClaimAGI`` false. Pure torch, deterministic.
"""
from __future__ import annotations

import torch


def walsh_hadamard_matrix(n: int, *, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Orthonormal ``n×n`` Walsh-Hadamard matrix (Sylvester construction); ``n`` a power of 2.

    Normalized by ``1/sqrt(n)`` so ``H Hᵀ = I``.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a positive power of 2, got {n}")
    H = torch.ones(1, 1, dtype=dtype, device=device)
    while H.shape[0] < n:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H / (n ** 0.5)


def random_hadamard_matrix(n: int, *, seed: int = 0, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Randomized Hadamard ``R = H · diag(±1)`` — orthonormal, deterministic from ``seed``.

    The random sign flip decorrelates the fixed Walsh pattern from the weight structure so a
    pathological weight can't align with ``H`` and defeat the outlier spreading.
    """
    H = walsh_hadamard_matrix(n, dtype=dtype, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=g).to(dtype=dtype, device=device) * 2 - 1)
    return H * signs.unsqueeze(0)  # H @ diag(signs), still orthonormal


def apply_input_rotation(W: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate the INPUT dimension of a linear weight: ``W -> W Rᵀ`` (for ``y = W x``).

    Paired with :func:`rotate_activations` (``x -> R x``) the linear is invariant:
    ``(W Rᵀ)(R x) = W x`` for orthonormal ``R``. ``W`` is ``[out, in]``, ``R`` is ``[in, in]``.
    """
    if W.shape[1] != R.shape[0]:
        raise ValueError(f"R must be [in,in]=[{W.shape[1]},{W.shape[1]}], got {tuple(R.shape)}")
    return (W.to(R.dtype) @ R.t()).to(W.dtype)


def rotate_activations(X: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate activations ``X`` ``[in, tokens]`` -> ``R X`` (the paired input rotation)."""
    if X.shape[0] != R.shape[0]:
        raise ValueError(f"R must be [in,in]=[{X.shape[0]},{X.shape[0]}], got {tuple(R.shape)}")
    return (R.to(X.dtype) @ X)


def block_absmax_stats(W: torch.Tensor, *, block: int = 16) -> dict:
    """Per-block-16 absmax distribution of ``W`` (what the NVFP4 micro-scale keys on).

    Lower ``maxScale`` / mean means fewer outlier-dominated blocks -> less FP4 error headroom.
    """
    flat = W.reshape(-1).float()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block)
    amax = blocks.abs().amax(dim=1)
    return {"maxBlockAbsmax": float(amax.max()), "meanBlockAbsmax": float(amax.mean()),
            "stdBlockAbsmax": float(amax.std())}


__all__ = [
    "walsh_hadamard_matrix",
    "random_hadamard_matrix",
    "apply_input_rotation",
    "rotate_activations",
    "block_absmax_stats",
]
