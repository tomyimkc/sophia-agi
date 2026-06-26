# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Numerically-correct reference kernels + roofline accounting (pure stdlib).

The first three fused ops on the kernel roadmap — softmax, RMSNorm, SwiGLU — as
exact, dependency-free reference implementations. They are the GROUND TRUTH a
Triton/CUDA kernel must match bit-for-bit (within tolerance), and their FLOP/byte
accounting feeds the existing roofline harness (`kernels/bench/roofline.py`) so we
can show — offline — that these ops are deeply memory-bound, which is exactly why
fusion (one HBM read/write instead of several) is the win.

References: Zhang & Sennrich 2019 (RMSNorm); Shazeer 2020 (GLU variants / SwiGLU);
Dao et al. 2022/2023 (FlashAttention IO-awareness); Williams et al. 2009 (roofline).
"""
from __future__ import annotations

import math

from kernels.bench.roofline import resolve_device


# --- exact reference ops -------------------------------------------------
def softmax(xs: "list[float]") -> "list[float]":
    """Numerically-stable row softmax (subtract max)."""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def silu(x: float) -> float:
    """SiLU / swish: x * sigmoid(x)."""
    return x / (1.0 + math.exp(-x))


def rmsnorm(xs: "list[float]", weight: "list[float]", *, eps: float = 1e-6) -> "list[float]":
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight."""
    if len(xs) != len(weight):
        raise ValueError("xs and weight length mismatch")
    ms = sum(x * x for x in xs) / len(xs)
    inv = 1.0 / math.sqrt(ms + eps)
    return [x * inv * w for x, w in zip(xs, weight)]


def swiglu(gate: "list[float]", up: "list[float]") -> "list[float]":
    """SwiGLU activation: silu(gate) * up (elementwise)."""
    if len(gate) != len(up):
        raise ValueError("gate and up length mismatch")
    return [silu(g) * u for g, u in zip(gate, up)]


# --- FLOP / byte accounting (the auditable roofline denominator) ---------
# FLOPs/element are small constants; the point is intensity << ridge => memory-bound.
_FLOPS_PER_ELEM = {"softmax": 5, "rmsnorm": 4, "swiglu": 6}


def op_cost(op: str, rows: int, d: int, *, dtype_bytes: int = 2) -> dict:
    """FLOPs, HBM bytes (one read of inputs + one write of output), and arithmetic
    intensity for an elementwise/reduction op over a (rows × d) tensor."""
    if op not in _FLOPS_PER_ELEM:
        raise ValueError(f"unknown op {op!r}; known: {sorted(_FLOPS_PER_ELEM)}")
    n = rows * d
    flops = _FLOPS_PER_ELEM[op] * n
    # inputs read + output written, each once if perfectly fused.
    n_inputs = 2 if op == "swiglu" else 1
    bytes_moved = dtype_bytes * (n_inputs * n + n)
    return {
        "op": op, "rows": rows, "d": d,
        "flops": flops, "bytes": bytes_moved,
        "intensity": flops / bytes_moved if bytes_moved else 0.0,
    }


def ridge_point(device: str, dtype: str = "bf16") -> float:
    """FLOP/byte where the compute & bandwidth ceilings cross. Ops with arithmetic
    intensity below this are memory-bound on that device."""
    spec = resolve_device(device)
    return spec.peak_flops(dtype) / spec.peak_bw()


def classify(op: str, rows: int, d: int, device: str, *, dtype: str = "bf16") -> dict:
    """Roofline regime for an op on a device: memory-bound iff intensity < ridge."""
    cost = op_cost(op, rows, d)
    ridge = ridge_point(device, dtype)
    cost["ridgePoint"] = round(ridge, 4)
    cost["regime"] = "memory-bound" if cost["intensity"] < ridge else "compute-bound"
    return cost
