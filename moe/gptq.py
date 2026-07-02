# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""GPTQ — Hessian-aware post-training quantization that minimizes OUTPUT error (输出误差最小化).

Round-to-nearest (RTN) quantizes each weight independently to minimize per-weight error.
But the served metric the low-RAM cert scores is *output* fidelity (next-token KL / top1),
and RTN throws away the cross-column structure that determines the output. GPTQ (Frantar,
Ashkboos, Hoefler, Alistarh 2022, arXiv:2210.17323) instead minimizes ``||(W - Wq) X||_F^2``
over a calibration input ``X`` by greedily quantizing column-by-column and *compensating*
the not-yet-quantized columns with the quantization error, weighted by the inverse Hessian
``H^-1`` where ``H = X X^T``.

This is the "PTQ finish" lever for the OLMoE NVFP4 cert: the v-series QAT co-adapts the
weights, but the final grid snap in ``certify_lowram.quantize_served_params`` is plain RTN
(``p.copy_(fq(p))``). Swapping that snap for a GPTQ snap against the SAME NVFP4 grid attacks
the residual top1 gap at its diagnosed root cause — a weight-MSE round that ignores output KL.

Pure ``torch``; deterministic; no training. ``canClaimAGI`` false — this is a serving-quality
mechanism, not a capability claim. The quantizer is injected so the SAME grid the cert serves
(NVFP4 via ``training.qat._torch_nvfp4``) is used, keeping the number served-reproducible.
"""
from __future__ import annotations

from typing import Callable

import torch

# A column quantizer maps a weight column (shape ``[rows]``) to its grid-snapped value.
ColumnQuantizer = Callable[[torch.Tensor], torch.Tensor]


def uniform_symmetric_quantizer(n_bits: int = 4) -> ColumnQuantizer:
    """Per-column absmax symmetric uniform quantizer to ``2^(n_bits-1)-1`` levels.

    Reference grid for tests and a generic INT-N snap. ``scale = max|col| / qmax``;
    ``q = clip(round(col/scale), -qmax, qmax) * scale``. A zero column maps to itself.
    """
    qmax = float(2 ** (n_bits - 1) - 1)

    def _q(col: torch.Tensor) -> torch.Tensor:
        amax = col.abs().max()
        if amax == 0:
            return col.clone()
        scale = amax / qmax
        return torch.clamp(torch.round(col / scale), -qmax, qmax) * scale

    return _q


def rtn_quantize(W: torch.Tensor, quantizer: ColumnQuantizer) -> torch.Tensor:
    """Round-to-nearest baseline: quantize each column independently (no compensation)."""
    Wq = torch.empty_like(W, dtype=torch.float32)
    Wf = W.float()
    for j in range(Wf.shape[1]):
        Wq[:, j] = quantizer(Wf[:, j])
    return Wq.to(dtype=W.dtype)


def gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    quantizer: ColumnQuantizer,
    *,
    blocksize: int = 128,
    percdamp: float = 0.01,
) -> torch.Tensor:
    """GPTQ quantize ``W`` [rows, cols] given Hessian ``H`` [cols, cols] = ``X X^T``.

    Greedy column-wise quantization with inverse-Hessian error compensation (Frantar et al.
    2022, arXiv:2210.17323), minimizing ``||(W - Wq) X||_F^2``. ``quantizer`` snaps one
    column at a time to the target grid (inject the NVFP4 grid for a served-reproducible
    cert, or :func:`uniform_symmetric_quantizer` for a generic INT-N snap).

    Deterministic. Dead columns (zero Hessian diagonal — never activated) are zeroed. A
    ``percdamp`` fraction of ``mean(diag H)`` is added to the diagonal for numerical
    stability of the Cholesky. Returns ``Wq`` in ``W``'s dtype.
    """
    if W.dim() != 2:
        raise ValueError(f"W must be 2-D [rows, cols], got shape {tuple(W.shape)}")
    rows, cols = W.shape
    if H.shape != (cols, cols):
        raise ValueError(f"H must be [cols, cols]=[{cols},{cols}], got {tuple(H.shape)}")

    Wf = W.clone().float()
    Hf = H.clone().float()

    dead = torch.diag(Hf) == 0
    Hf[dead, dead] = 1.0
    Wf[:, dead] = 0.0

    damp = percdamp * torch.mean(torch.diag(Hf))
    idx = torch.arange(cols, device=Wf.device)
    Hf[idx, idx] += damp

    # Upper-triangular Cholesky factor of H^-1 (the GPTQ update operator).
    L = torch.linalg.cholesky(Hf)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Wq = torch.zeros_like(Wf)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1
        W1 = Wf[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = quantizer(w)
            Q1[:, i] = q
            err = (w - q) / d
            # compensate the remaining columns IN this block
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Wq[:, i1:i2] = Q1
        # propagate the accumulated block error to the columns AFTER this block
        Wf[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    Wq[:, dead] = 0.0
    return Wq.to(dtype=W.dtype)


def output_mse(W: torch.Tensor, Wq: torch.Tensor, X: torch.Tensor) -> float:
    """Mean squared *output* error ``mean(((W - Wq) X)^2)`` — the quantity GPTQ minimizes.

    ``X`` is ``[cols, n_samples]`` (the calibration activations whose ``X X^T`` is ``H``).
    """
    d = (W.float() - Wq.float()) @ X.float()
    return float((d * d).mean())


__all__ = [
    "ColumnQuantizer",
    "uniform_symmetric_quantizer",
    "rtn_quantize",
    "gptq_quantize",
    "output_mse",
]
