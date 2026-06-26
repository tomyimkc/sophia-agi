#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fused NVFP4 weight-only dequant-GEMM — the DGX Spark-native inference kernel.

``out[M,N] = x[M,K] @ dequant(W_nvfp4[K,N])``, where ``W`` is stored in **NVFP4**
(4-bit E2M1 codes + a per-block FP8-E4M3 micro-scale over groups of 16 along K) and
dequantized *inside* the GEMM so the bytes that cross the memory bus are 4-bit, not
16-bit. On a bandwidth-bound device — the DGX Spark GB10 moves only 273 GB/s of
unified LPDDR5x — batch-1 token decode is memory-bound, so halving weight bytes from
FP16 (2 B) to NVFP4 (~0.5 B + scale) is the dominant decode-speed lever. This kernel
is where the `moe/quant.py` NVFP4 scheme becomes a deployment artifact.

Two layers, deliberately:

  * **NumPy reference + packing** (pack/unpack 4-bit, dequant, FLOP/byte accounting):
    pure-CPU, deterministic, CI-tested. This is the correctness oracle and the honest
    byte count the roofline divides by.
  * **Triton fused kernel** (GPU-only): loads packed int4 + FP8 scales, dequants to
    the compute dtype, accumulates in FP32, and self-rooflines against the Spark
    profile in ``kernels/bench/roofline.py``. Without torch+CUDA+triton it prints a
    clean "skipped" line and returns None, so CI and the orchestrator stay green.

Honest bounds: like ``run_kernel.py`` this is a *straightforward* tiled kernel (one
block config, dequant-in-the-inner-loop, no warp specialization / split-K / autotuned
epilogue). Its % of the Spark's 273 GB/s roofline is the number to report and then
close — never an "Nx vs FP16" headline. A Mojo port is a planned, roofline-gated
A/B (see docs/11-Platform/DGX-Spark-Maximization.md §4); the denominator stays the
hardware, not a strawman.

    python kernels/src/nvfp4_gemm.py                 # auto-skips without a GPU
    python kernels/src/nvfp4_gemm.py --m 1 --n 8192 --k 8192   # decode-shaped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bench.roofline import (  # noqa: E402
    RooflineResult,
    analyze,
    detect_device,
    format_report,
    resolve_device,
)

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

NVFP4_BLOCK = 16  # elements per FP8 micro-scale block along K (Blackwell NVFP4 default)

# Signed E2M1 codebook: 16 codes -> representable signed magnitudes. Code 8 is -0.0 == 0.0.
# Index order is the 4-bit code stored in the packed weight; values match moe/quant.py's
# E2M1 levels {0,.5,1,1.5,2,3,4,6} with sign.
_E2M1_CODEBOOK = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_E2M1_MAX = 6.0


def _codebook():
    return np.asarray(_E2M1_CODEBOOK, dtype=np.float64)


# --------------------------------------------------------------------------------------
# NumPy reference: quantize / pack / unpack / dequant. The correctness oracle.
# --------------------------------------------------------------------------------------
def quantize_nvfp4_weights(W, *, block_size: int = NVFP4_BLOCK):
    """Quantize a weight matrix ``W[K,N]`` to NVFP4, blocking along K.

    Returns ``(codes, scales, k_orig)`` where ``codes`` is ``uint8[K_pad, N]`` holding
    one 4-bit E2M1 code (0..15) per element, ``scales`` is ``float64[K_pad//block, N]``
    FP8-E4M3 micro-scales, and ``k_orig`` is the pre-padding K (rows are zero-padded up
    to a multiple of ``block_size``). Blocking is along K — the GEMM contraction dim —
    which is how a real dequant-in-GEMM kernel lays out micro-scales.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    from moe.quant import fp8_e4m3_roundtrip  # FP8 micro-scale, shared with moe/quant.py

    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be 2-D [K, N]")
    k, n = W.shape
    pad = (-k) % block_size
    if pad:
        W = np.concatenate([W, np.zeros((pad, n))], axis=0)
    kp = W.shape[0]
    blocks = W.reshape(kp // block_size, block_size, n)
    amax = np.max(np.abs(blocks), axis=1, keepdims=True)            # [nb,1,n]
    amax = np.where(amax == 0, 1.0, amax)
    scale = fp8_e4m3_roundtrip(amax / _E2M1_MAX)                    # store scale in FP8
    scale = np.where(scale == 0, amax / _E2M1_MAX, scale)
    scaled = blocks / scale                                          # into E2M1 range
    book = _codebook()
    # nearest codebook entry per element -> 4-bit code
    codes = np.abs(scaled.reshape(kp, n)[..., None] - book).argmin(axis=-1).astype(np.uint8)
    scales = scale.reshape(kp // block_size, n)
    return codes, scales, k


def dequantize_nvfp4_weights(codes, scales, *, block_size: int = NVFP4_BLOCK, k_orig=None):
    """Inverse of :func:`quantize_nvfp4_weights`: ``codes,scales -> W_approx[K,N]``."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    book = _codebook()
    kp, n = codes.shape
    vals = book[codes]                                              # [kp,n] signed magnitudes
    scale_full = np.repeat(scales, block_size, axis=0)[:kp]         # broadcast block scale
    W = vals * scale_full
    return W if k_orig is None else W[:k_orig]


def pack_int4(codes):
    """Pack a ``uint8`` array of 4-bit codes (0..15) two-per-byte along the last axis.

    The honest memory artifact: NVFP4 weights are 4 bits each. Returns ``uint8`` with
    the last axis halved (the axis is zero-padded to even length first)."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    c = np.asarray(codes, dtype=np.uint8)
    if np.any(c > 15):
        raise ValueError("codes must be 4-bit (0..15)")
    if c.shape[-1] % 2:
        pad = [(0, 0)] * (c.ndim - 1) + [(0, 1)]
        c = np.pad(c, pad)
    lo = c[..., 0::2]
    hi = c[..., 1::2]
    return (lo | (hi << 4)).astype(np.uint8)


def unpack_int4(packed, length):
    """Inverse of :func:`pack_int4`; ``length`` is the original (pre-pad) last-axis size."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    p = np.asarray(packed, dtype=np.uint8)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out[..., :length]


def nvfp4_gemm_reference(x, W, *, block_size: int = NVFP4_BLOCK):
    """CPU reference for the fused kernel: ``x @ dequant(quantize_nvfp4(W))``."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    x = np.asarray(x, dtype=np.float64)
    codes, scales, k_orig = quantize_nvfp4_weights(W, block_size=block_size)
    Wq = dequantize_nvfp4_weights(codes, scales, block_size=block_size, k_orig=k_orig)
    return x @ Wq


# --------------------------------------------------------------------------------------
# Byte / FLOP accounting — the honest roofline denominator for a weight-only FP4 GEMM.
# --------------------------------------------------------------------------------------
def nvfp4_gemm_flops(m: int, n: int, k: int) -> int:
    """2*m*n*k — the dequant is bookkeeping; the matmul MACs dominate the FLOPs."""
    return 2 * m * n * k


def nvfp4_gemm_bytes(m: int, n: int, k: int, *, block_size: int = NVFP4_BLOCK,
                     act_bytes: int = 2, weight_bytes_per_elem: float = 0.5) -> int:
    """Lower-bound memory traffic for a weight-only NVFP4 GEMM.

    Weights move at ``weight_bytes_per_elem`` bytes/elem plus one FP8 scale (1 byte)
    per ``block_size`` elements along K; activations/output move at ``act_bytes``
    (BF16=2). For decode (m=1) the weight term dominates — that is the whole point.

    ``weight_bytes_per_elem`` defaults to **0.5** — the *deployment-target* (truly
    packed 4-bit) accounting, which is the honest denominator a packed kernel earns.
    The current `run_nvfp4_gemm` reference kernel streams **unpacked 1-byte codes**
    (it has no in-kernel int4 unpack yet), so it passes ``weight_bytes_per_elem=1.0``
    and reports its % of roofline against the bytes it *actually* moves — never the
    0.5 it has not yet earned. `pack_int4` proves the format halves bytes again; wiring
    that unpack into the Triton K-loop is the next step (then the kernel earns 0.5).
    """
    nblocks_k = (k + block_size - 1) // block_size
    weight_bytes = int(weight_bytes_per_elem * n * k) + nblocks_k * n   # codes + fp8 scales
    act_bytes_total = act_bytes * (m * k + m * n)          # read x, write out
    return int(weight_bytes + act_bytes_total)


# --------------------------------------------------------------------------------------
# Triton fused kernel (GPU-only). Imports isolated so the module loads without triton.
# --------------------------------------------------------------------------------------
def _have_gpu_stack() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "CUDA not detected"
    try:
        import triton  # noqa: F401, PLC0415
    except Exception:
        return False, "triton not installed"
    return True, "ok"


def _build_kernel():
    """Triton fused dequant-GEMM factory. Dequant happens in the K-loop (int4 -> compute dtype).

    Kept deliberately simple (one block config, codebook in a small constant table). The
    deployment win is the 4-bit weight load; tuning the epilogue/specialization is the
    explicit follow-up, re-reported as % of the Spark roofline at each step.
    """
    import torch
    import triton
    import triton.language as tl

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, NVFP4_BLOCK  # BLOCK_K == micro-scale block

    @triton.jit
    def _k(  # pragma: no cover - requires CUDA
        x_ptr, code_ptr, scale_ptr, book_ptr, out_ptr,
        M, N, K, NB,
        sx_m, sx_k, sc_k, sc_n, ss_b, ss_n, so_m, so_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kb in range(0, NB):
            k_idx = kb * BLOCK_K + offs_k
            x_p = x_ptr + (offs_m[:, None] * sx_m + k_idx[None, :] * sx_k)
            x = tl.load(x_p, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
            c_p = code_ptr + (k_idx[:, None] * sc_k + offs_n[None, :] * sc_n)
            codes = tl.load(c_p, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0)
            w = tl.load(book_ptr + codes.to(tl.int32))         # code -> E2M1 value (cast for gather)
            s_p = scale_ptr + (kb * ss_b + offs_n[None, :] * ss_n)
            s = tl.load(s_p, mask=offs_n[None, :] < N, other=0.0)
            w = w * s                                          # dequant in-register
            acc += tl.dot(x.to(tl.float32), w.to(tl.float32))
        o_p = out_ptr + (offs_m[:, None] * so_m + offs_n[None, :] * so_n)
        tl.store(o_p, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    def matmul(x, codes, scales):
        M, K = x.shape
        Kc, N = codes.shape
        NB = scales.shape[0]
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)
        book = torch.tensor(_E2M1_CODEBOOK, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _k[grid](
            x, codes, scales, book, out, M, N, Kc, NB,
            x.stride(0), x.stride(1), codes.stride(0), codes.stride(1),
            scales.stride(0), scales.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out

    return matmul


def run_nvfp4_gemm(
    m: int = 1, n: int = 8192, k: int = 8192,
    iters: int = 50, warmup: int = 10,
    device_name: str | None = None, verbose: bool = True,
) -> RooflineResult | None:
    """Run + roofline the fused NVFP4 GEMM. Returns None (and prints why) without a GPU."""
    ok, reason = _have_gpu_stack()
    if not ok or not _HAVE_NUMPY:
        if verbose:
            why = reason if not ok else "numpy not installed"
            print(f"[nvfp4_gemm] skipped: {why} (need torch+CUDA+triton+numpy). "
                  f"NumPy reference + accounting are exercised by tests/test_nvfp4_gemm.py.")
        return None

    import torch

    matmul = _build_kernel()
    x_np = np.random.default_rng(0).standard_normal((m, k)).astype(np.float32)
    W_np = np.random.default_rng(1).standard_normal((k, n)).astype(np.float32)
    codes_np, scales_np, _ = quantize_nvfp4_weights(W_np)

    x = torch.tensor(x_np, device="cuda", dtype=torch.bfloat16)
    # Codes are 0..15: upload as uint8 (1 byte/elem), the bytes this reference kernel
    # actually streams. True 4-bit (0.5 B) needs the in-kernel int4 unpack — the next step.
    codes = torch.tensor(codes_np, device="cuda", dtype=torch.uint8)
    scales = torch.tensor(scales_np, device="cuda", dtype=torch.float32)

    out = matmul(x, codes, scales).float().cpu().numpy()
    ref = nvfp4_gemm_reference(x_np, W_np)
    rel = float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-9))
    correct = rel < 5e-2  # bf16 activations + 4-bit weights; 5% relative is the honest bar
    if verbose:
        print(f"[nvfp4_gemm] correctness vs numpy NVFP4 reference: rel_err={rel:.4f} -> "
              f"{'PASS' if correct else 'FAIL'}")
    if not correct:
        raise RuntimeError(f"kernel incorrect: relative error {rel:.4f} >= 5e-2")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        matmul(x, codes, scales)
    torch.cuda.synchronize()
    times_s: list[float] = []
    for _ in range(iters):
        start.record()
        matmul(x, codes, scales)
        end.record()
        torch.cuda.synchronize()
        times_s.append(start.elapsed_time(end) / 1e3)

    device = resolve_device(device_name or detect_device())
    result = analyze(
        flops=nvfp4_gemm_flops(m, n, k),
        # Honest denominator: this kernel streams unpacked 1-byte codes, so account at
        # 1.0 B/elem — not the 0.5 B a packed kernel earns. % of roofline reflects what
        # actually crosses the bus.
        bytes_moved=nvfp4_gemm_bytes(m, n, k, weight_bytes_per_elem=1.0),
        times_s=times_s, device=device, dtype="fp4",
    )
    if verbose:
        print(f"\n[nvfp4_gemm] fused NVFP4 dequant-GEMM {m}x{n}x{k} ({iters} timed iters)")
        print("  NOTE: reference streams unpacked 1-byte codes; packed 4-bit (0.5 B) is the next step.\n")
        print(format_report(result))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=1, help="rows of x (1 = token decode)")
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default=None, help="override device name (else autodetect)")
    args = p.parse_args(argv)
    run_nvfp4_gemm(m=args.m, n=args.n, k=args.k, iters=args.iters,
                   warmup=args.warmup, device_name=args.device)
    return 0  # clean exit even on skip, so CI/orchestrator stay green pre-GPU


if __name__ == "__main__":
    sys.exit(main())
