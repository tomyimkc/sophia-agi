# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fused row-softmax Triton kernel — GPU-only (DGX Spark / RunPod).

A single-pass, numerically-stable row softmax: load each row once into SRAM,
compute max → exp → sum → divide, write once. One HBM read + one write instead of
the several passes an unfused eager implementation makes — the memory-bound win
`kernels/reference.op_cost('softmax', ...)` predicts offline.

NOT imported by CI (it `import triton` at module load). Run on a CUDA device;
validate against `kernels.reference.softmax` and bench with `kernels/bench/roofline.py`.
"""
from __future__ import annotations

import triton  # noqa: F401  (GPU-only; absent in CI by design)
import triton.language as tl


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, row_stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    ptrs = x_ptr + row * row_stride + cols
    x = tl.load(ptrs, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)            # numerical stability
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom
    tl.store(out_ptr + row * row_stride + cols, y, mask=mask)


def softmax(x):
    """Row-softmax of a 2D CUDA tensor `x` (returns a new tensor). Requires torch+triton."""
    import torch  # local import; GPU path only

    assert x.is_cuda and x.dim() == 2, "expected a 2D CUDA tensor"
    rows, cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(cols)
    _softmax_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK=block)
    return out
