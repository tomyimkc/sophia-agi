# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia kernels — operator-level building blocks for the systems track.

Reproductions of the kernels that make large-model train/inference tractable,
written reference-first: a dependency-light numpy implementation that is proven
*numerically identical* to the textbook formula in CI, plus an optional fused
GPU kernel (Triton) behind a gated import for when a CUDA device is present.

- ``flash_attention`` : FlashAttention-style tiled, online-softmax attention.
  Reproduces Dao et al. 2022 (arXiv:2205.14135): the same output as O(N²)-memory
  softmax attention while only ever materializing one ``Bq×Bk`` score tile — the
  algorithm that makes long-context attention fit in SRAM. Numpy reference is
  CI-tested for exact agreement; the Triton kernel is gated.
- ``indexshare`` : IndexShare — reusing a sparse-attention index across layers
  (the GLM-5.2 attention innovation). A numpy reproduction of the amortization
  principle with a measured quality-vs-compute curve, CI-tested for the
  index-computed-once invariant and bounded sharing error.

See ``docs/SYSTEMS-TRACK.md`` for how this maps to the role's "CUDA/Triton 算子
开发"、"长上下文" and "复现论文" lines, and ``docs/11-Platform/Cheap-Compute-Boundary.md``
for the honest scope.
"""

from __future__ import annotations

from kernels.flash_attention import (
    flash_attention_numpy,
    naive_attention,
    triton_available,
)
from kernels.indexshare import (
    build_index,
    indexshare_block,
    per_layer_baseline,
    quality_vs_compute_curve,
    sparse_attention_indexed,
)

__all__ = [
    "flash_attention_numpy",
    "naive_attention",
    "triton_available",
    # IndexShare (cross-layer index amortization)
    "build_index",
    "sparse_attention_indexed",
    "indexshare_block",
    "per_layer_baseline",
    "quality_vs_compute_curve",
]
