#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""First real M1 kernel: a Triton tiled BF16 GEMM, reported against the roofline.

C[M,N] = A[M,K] @ B[K,N], BF16 inputs, FP32 accumulate (tensor-core ``tl.dot``).

This is the kernel ``tools/runpod_kernels.py`` looks for. On a CUDA GPU it:
  1. checks correctness against ``torch.matmul`` (BF16 tolerance),
  2. times >=3 trials with CUDA events (warmup excluded),
  3. prints its own ``roofline.analyze(...)`` block — % of the physical limit, honest.

Honest bounds: this is a *straightforward* tiled GEMM (single fixed block config, no
warp specialization, no split-K, no autotuned epilogue). It will NOT hit cuBLAS/CUTLASS
% of peak — and that gap, in % of roofline, is exactly the M1 number worth reporting and
then closing. No "Nx vs naive" framing: the denominator is the hardware, not a strawman.

Offline/CI-safe: with no CUDA or no Triton it prints a clear "skipped" line and exits 0,
so the orchestrator and tests stay green before a GPU is rented.

    python kernels/src/run_kernel.py                       # auto-skips without a GPU
    python kernels/src/run_kernel.py --m 4096 --n 4096 --k 4096 --iters 50
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
    gemm_bytes,
    gemm_flops,
    resolve_device,
)


def _have_gpu_stack() -> tuple[bool, str]:
    """Return (ok, reason). ok only if torch+CUDA+triton are all present."""
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
    """Import-time-isolated Triton kernel factory (so the module imports without triton)."""
    import triton
    import triton.language as tl

    # Fixed, conservative block config. Autotuning is an explicit M1+ follow-up.
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M = 128, 128, 64, 8

    @triton.jit
    def _gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # Grouped (super-block) program ordering for better L2 reuse.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc += tl.dot(a, b)  # BF16 inputs -> FP32 accumulate on tensor cores
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = acc.to(tl.float16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def matmul(a, b):
        import torch

        M, K = a.shape
        K2, N = b.shape
        assert K == K2, "inner dims must match"
        c = torch.empty((M, N), device=a.device, dtype=torch.float16)
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        _gemm_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        )
        return c

    return matmul


def run_gemm(
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    iters: int = 50,
    warmup: int = 10,
    dtype: str = "bf16",
    device_name: str | None = None,
    verbose: bool = True,
) -> RooflineResult | None:
    """Run + roofline the Triton GEMM. Returns None (and prints why) if no GPU stack."""
    ok, reason = _have_gpu_stack()
    if not ok:
        if verbose:
            print(f"[run_kernel] skipped: {reason} (need torch+CUDA+triton). "
                  f"Roofline gate is exercised by kernels/bench/roofline.py --self-test.")
        return None

    import torch

    matmul = _build_kernel()
    # Explicit, unambiguous mapping: "half"/"float16" mean FP16, not BF16. run_gemm is a
    # public entry point, so validate rather than silently coercing an unknown dtype.
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float16": torch.float16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {dtype!r}; use one of {sorted(dtype_map)}")
    torch_dtype = dtype_map[dtype]
    a = torch.randn((m, k), device="cuda", dtype=torch_dtype)
    b = torch.randn((k, n), device="cuda", dtype=torch_dtype)

    # Correctness vs torch reference (compute the reference in fp32 for a fair tolerance).
    c_tri = matmul(a, b).float()
    c_ref = (a.float() @ b.float())
    max_abs = (c_tri - c_ref).abs().max().item()
    rel = max_abs / (c_ref.abs().max().item() + 1e-9)
    correct = rel < 2e-2  # BF16 GEMM accumulates error; 2% relative is the honest bar
    if verbose:
        print(f"[run_kernel] correctness vs torch: max_rel_err={rel:.4f} -> "
              f"{'PASS' if correct else 'FAIL'}")
    if not correct:
        raise RuntimeError(f"kernel incorrect: relative error {rel:.4f} >= 2e-2")

    # Timed trials with CUDA events; warmup excluded.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        matmul(a, b)
    torch.cuda.synchronize()

    times_s: list[float] = []
    for _ in range(iters):
        start.record()
        matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        times_s.append(start.elapsed_time(end) / 1e3)  # ms -> s

    device = resolve_device(device_name or detect_device())
    flops = gemm_flops(m, n, k)
    # BF16 = 2 bytes; the lower-bound HBM traffic (each of A,B,C touched once).
    dtype_bytes = 2
    bytes_moved = gemm_bytes(m, n, k, dtype_bytes)

    result = analyze(flops=flops, bytes_moved=bytes_moved, times_s=times_s,
                     device=device, dtype=dtype)
    if verbose:
        print(f"\n[run_kernel] Triton tiled GEMM {m}x{n}x{k} {dtype} "
              f"({iters} timed iters)\n")
        print(format_report(result))
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--device", default=None, help="override device name (else autodetect)")
    args = p.parse_args(argv)
    # Always exit 0 on a clean skip so CI / the orchestrator stay green pre-GPU.
    run_gemm(m=args.m, n=args.n, k=args.k, iters=args.iters, warmup=args.warmup,
             dtype=args.dtype, device_name=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
