#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Triton GEMM kernel — CPU/offline-safe behavior.

The real timing path needs a CUDA GPU + Triton (covered on the pod via
tools/runpod_kernels.py). On CPU/CI we verify the module imports without torch/triton and
that the run path skips cleanly (returns None, exit 0) rather than crashing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.src.run_kernel import _have_gpu_stack, main, run_gemm  # noqa: E402


def test_imports_without_torch_or_triton():
    # Importing the module (done above) must not require torch/triton.
    ok, reason = _have_gpu_stack()
    # In CI there is no GPU stack; assert the gate reports that honestly.
    if not ok:
        assert reason in {"torch not installed", "CUDA not detected", "triton not installed"}


def test_run_gemm_skips_cleanly_without_gpu():
    ok, _ = _have_gpu_stack()
    if ok:  # on a real GPU box this test is a no-op (the live path is pod-only)
        return
    assert run_gemm(m=256, n=256, k=256, iters=2, verbose=False) is None


def test_main_exits_zero_on_skip():
    ok, _ = _have_gpu_stack()
    if ok:
        return
    assert main(["--m", "256", "--n", "256", "--k", "256", "--iters", "2"]) == 0
