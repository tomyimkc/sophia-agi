# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia MoE + low-precision — the two levers behind DeepSeek-scale efficiency.

- ``router``  : top-k Mixture-of-Experts routing — softmax gating, capacity-bounded
  dispatch/combine, and the Switch-Transformer load-balancing auxiliary loss that
  keeps experts evenly utilised. This is the sparsity that lets a model have huge
  parameter count at a fraction of the active FLOPs (GShard / Switch / DeepSeekMoE).
- ``quant``   : low-precision weight quantization — symmetric INT8 (per-tensor and
  per-channel) and an FP8-E4M3 emulation, with round-trip error bounds and a
  weight-only quantized linear. The "低精度训推" lever for memory + bandwidth.
- ``adapt``   : adaptive (mixed-precision) quantization — the Unsloth-Dynamic-2.0
  core idea (per-tensor bit allocation by measured output-KL sensitivity) made
  reproducible and governed: a greedy budgeted allocator with a protected floor
  for critical tensors (embeddings, lm_head, routers), CI-checked.
- ``calibrate``: calibration-distribution matching — source the quantizer's
  calibration set from the deployment distribution (chat-formatted) and prove it
  is decontaminated from eval, reusing the repo's contamination guard.

Reference implementations in numpy, proven against their defining equations in
CI. See ``docs/SYSTEMS-TRACK.md`` and ``docs/11-Platform/Cheap-Compute-Boundary.md``.
"""

from __future__ import annotations

from moe.adapt import (
    PROTECTED_FLOOR,
    avg_bits,
    bit_allocator,
    kl_divergence,
    measure_sensitivity,
    quantize_uniform,
    total_bits,
)
from moe.calibrate import (
    build_calibration_set,
    calibration_datasheet,
    check_calibration_disjoint,
)
from moe.quant import (
    dequantize_int8,
    fp8_e4m3_roundtrip,
    quantize_int8,
    quantized_linear,
)
from moe.router import MoERouter, load_balancing_loss, top_k_gating

__all__ = [
    "MoERouter",
    "top_k_gating",
    "load_balancing_loss",
    "quantize_int8",
    "dequantize_int8",
    "quantized_linear",
    "fp8_e4m3_roundtrip",
    # adaptive mixed-precision
    "bit_allocator",
    "measure_sensitivity",
    "quantize_uniform",
    "kl_divergence",
    "total_bits",
    "avg_bits",
    "PROTECTED_FLOOR",
    # calibration matching
    "build_calibration_set",
    "check_calibration_disjoint",
    "calibration_datasheet",
]
