# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Triton GPU kernels (run on the DGX Spark / RunPod — NOT imported in CI).

Each module here `import triton` at top, so it loads ONLY on a CUDA box. The
offline core (`kernels/reference.py`, `tools/run_kernels.py --mode mock`, and the
CI tests) never imports this package — it validates the reference numerics and the
roofline accounting without a GPU. On GPU, a kernel is correct iff it matches
`kernels.reference` within tolerance, and is benchmarked against
`torch.nn.functional` via `kernels/bench/roofline.py` (% of HBM SOL, ncu-attributed).
"""
