#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gather-on-read-set NVFP4 GEMM — the Tier-2 bandwidth lever for Governed Speculative Sparsity.

Tier 0/1 established that a real MoE reads only ρ≈0.10 of its expert weights per token and
that pruning to that read-set carries a measurable, bounded error. This kernel turns that
into **bytes that never cross the bus**: the contraction dim ``K`` is split into tiles (an
expert, or a channel group); a per-step **read-set mask** selects a subset; the GEMM reads
and computes **only the selected tiles**.

    dense :   out = x[M,K]      @ dequant(W_nvfp4[K,N])
    gather:   out = x[M,K_sel]  @ dequant(W_nvfp4[K_sel,N])   ,  K_sel = ρ·K

On the bandwidth-bound Spark (273 GB/s unified LPDDR5x) decode time ≈ bytes_read / BW, so
reading ``ρ·K`` tiles instead of ``K`` is a **ρ× weight-traffic** reduction — the whole point
on a machine where you can't hide transfer, only avoid it.

Two layers, like ``nvfp4_gemm.py``:
  * **NumPy reference + byte accounting** (pure-CPU, CI-tested): the gather GEMM and the
    honest roofline denominator. Correctness oracle: a *full* mask reproduces the dense NVFP4
    GEMM exactly; a partial mask equals the dense GEMM with the non-selected K-rows zeroed.
  * **GPU path** (skips cleanly without CUDA): **gathers the selected tiles into compacted
    buffers and reuses the validated dense kernel** from ``nvfp4_gemm.py`` — so the novelty
    (the gather + the byte accounting) is isolated and the matmul stays correct-by-construction.
    Self-rooflines against the Spark profile; ``rel < 5e-2`` vs the NumPy reference.

Honest scope: this measures the **bandwidth** GSS saves (bytes-read at ρ), the aggressive
(pruned-verify) lever from `Real-Tensor-Movement-Thesis.md` §4.3. The *output* it computes is
the pruned forward — exact only over the selected tiles — so a lossless decode still pairs it
with the dense-verify accept/reject (`serving/gss.py`). Report **% of the Spark roofline**,
never "Nx vs a strawman". `canClaimAGI` stays `false`.

    python kernels/src/gss_gather_gemm.py                         # auto-skips without a GPU
    python kernels/src/gss_gather_gemm.py --m 1 --n 8192 --k 8192 --tile 256 --rho 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bench.roofline import (  # noqa: E402
    RooflineResult, analyze, detect_device, format_report, resolve_device,
)
from kernels.src.nvfp4_gemm import (  # noqa: E402
    NVFP4_BLOCK, dequantize_nvfp4_weights, quantize_nvfp4_weights,
)

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


# --------------------------------------------------------------------------------------
# NumPy reference + byte accounting — the correctness oracle and roofline denominator.
# --------------------------------------------------------------------------------------

def n_tiles(k: int, tile_size: int) -> int:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    return (k + tile_size - 1) // tile_size


def gather_gemm_reference(x, W, tile_mask, *, tile_size: int, block_size: int = NVFP4_BLOCK):
    """``out = x @ dequant(W_nvfp4)`` reading only the K-tiles where ``tile_mask`` is True.

    ``x`` [M,K], ``W`` [K,N]; ``tile_mask`` is length ``ceil(K/tile_size)``. Weights are NVFP4
    (E2M1 + FP8 micro-scale, the `nvfp4_gemm` scheme). A full mask == the dense NVFP4 GEMM.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if x.ndim != 2 or W.ndim != 2:
        raise ValueError("x and W must be 2-D")
    if x.shape[1] != W.shape[0]:
        raise ValueError(f"contraction mismatch: x K={x.shape[1]} vs W K={W.shape[0]}")
    k, nmask = W.shape[0], n_tiles(W.shape[0], tile_size)
    mask = np.asarray(tile_mask, dtype=bool).ravel()
    if mask.shape[0] != nmask:
        raise ValueError(f"tile_mask len {mask.shape[0]} != n_tiles {nmask}")
    codes, scales, k_orig = quantize_nvfp4_weights(W, block_size=block_size)
    Wq = dequantize_nvfp4_weights(codes, scales, block_size=block_size, k_orig=k_orig)  # [K,N]
    out = np.zeros((x.shape[0], W.shape[1]), dtype=np.float64)
    for t in range(nmask):
        if not mask[t]:
            continue
        ksl = slice(t * tile_size, min((t + 1) * tile_size, k))
        out += x[:, ksl] @ Wq[ksl, :]
    return out


def gather_gemm_bytes(m: int, n: int, k: int, *, tile_size: int, n_selected: int,
                      block_size: int = NVFP4_BLOCK, act_bytes: int = 2,
                      weight_bytes_per_elem: float = 0.5) -> int:
    """Memory traffic when only ``n_selected`` of ``ceil(k/tile_size)`` K-tiles are read.

    Weights at 0.5 B/elem (packed NVFP4) + one FP8 scale per ``block_size`` along K, counted
    **only for the selected tiles**; activations read just the selected K columns of ``x``;
    the output is written once. With ``n_selected == n_tiles`` this equals the dense NVFP4
    traffic, so ``gather/dense ≈ ρ`` on the weight-dominated decode (m=1) path — the lever.
    """
    nt = n_tiles(k, tile_size)
    sel = max(0, min(n_selected, nt))
    k_sel = min(sel * tile_size, k)
    nblocks_k = (k_sel + block_size - 1) // block_size
    weight_bytes = int(weight_bytes_per_elem * n * k_sel) + nblocks_k * n
    act_bytes_total = act_bytes * (m * k_sel + m * n)
    return int(weight_bytes + act_bytes_total)


def gather_gemm_flops(m: int, n: int, k: int, *, tile_size: int, n_selected: int) -> int:
    """2·m·n·K_sel — only the selected tiles contract."""
    nt = n_tiles(k, tile_size)
    k_sel = min(max(0, min(n_selected, nt)) * tile_size, k)
    return 2 * m * n * k_sel


def read_set_tiles(rho: float, k: int, tile_size: int, *, seed: int = 0):
    """A boolean tile mask selecting ~``rho`` of the K-tiles (the read-set), seeded."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must be in (0, 1]")
    nt = n_tiles(k, tile_size)
    sel = max(1, int(round(rho * nt)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(nt, size=sel, replace=False)
    mask = np.zeros(nt, dtype=bool)
    mask[idx] = True
    return mask


# --------------------------------------------------------------------------------------
# GPU path — gather selected tiles into compacted buffers, reuse the validated dense kernel.
# --------------------------------------------------------------------------------------

def _have_gpu_stack() -> "tuple[bool, str]":
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


def run_gss_gather_gemm(m: int = 1, n: int = 8192, k: int = 8192, tile_size: int = 256,
                        rho: float = 0.10, iters: int = 50, warmup: int = 10,
                        device_name: "str | None" = None, seed: int = 0,
                        verbose: bool = True) -> "RooflineResult | None":
    """Gather the read-set tiles, run the dense NVFP4 kernel on the compacted problem, roofline it.

    The compacted GEMM is ``x[:, K_sel] @ dequant(W_nvfp4[K_sel, :])`` — reusing the validated
    `nvfp4_gemm` kernel, so correctness is inherited and only the gather is new. Returns None
    (and prints why) without a GPU. The roofline divides by `gather_gemm_bytes` (the ρ traffic).
    """
    ok, reason = _have_gpu_stack()
    if not ok or not _HAVE_NUMPY:
        if verbose:
            why = reason if not ok else "numpy not installed"
            print(f"[gss_gather_gemm] skipped: {why} (need torch+CUDA+triton+numpy). "
                  f"NumPy reference + accounting are exercised by tests/test_gss_gather_gemm.py.")
        return None

    import torch
    from kernels.src.nvfp4_gemm import _build_kernel, pack_int4_k

    rng = np.random.default_rng(seed)
    x_np = rng.standard_normal((m, k)).astype(np.float32)
    W_np = rng.standard_normal((k, n)).astype(np.float32)
    mask = read_set_tiles(rho, k, tile_size, seed=seed)
    sel_tiles = np.flatnonzero(mask)
    # Compact the selected K-tiles (the gather): only these columns of x / rows of W move.
    cols = np.concatenate([np.arange(t * tile_size, min((t + 1) * tile_size, k)) for t in sel_tiles])
    x_sel = np.ascontiguousarray(x_np[:, cols])
    W_sel = np.ascontiguousarray(W_np[cols, :])

    codes, scales, _ = quantize_nvfp4_weights(W_sel)
    packed_np = pack_int4_k(codes)
    matmul = _build_kernel()
    x = torch.tensor(x_sel, device="cuda", dtype=torch.bfloat16)
    packed = torch.tensor(packed_np, device="cuda", dtype=torch.uint8)
    scales_t = torch.tensor(scales, device="cuda", dtype=torch.float32)

    out = matmul(x, packed, scales_t).float().cpu().numpy()
    ref = gather_gemm_reference(x_np, W_np, mask, tile_size=tile_size)
    rel = float(np.linalg.norm(out - ref) / (np.linalg.norm(ref) + 1e-9))
    correct = rel < 5e-2
    if verbose:
        print(f"[gss_gather_gemm] ρ={rho:.3f} ({len(sel_tiles)}/{n_tiles(k, tile_size)} tiles); "
              f"correctness vs NumPy gather reference: rel_err={rel:.4f} -> {'PASS' if correct else 'FAIL'}")
    if not correct:
        raise RuntimeError(f"gather kernel incorrect: rel {rel:.4f} >= 5e-2")

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        matmul(x, packed, scales_t)
    torch.cuda.synchronize()
    times_s: list[float] = []
    for _ in range(iters):
        start.record(); matmul(x, packed, scales_t); end.record()
        torch.cuda.synchronize(); times_s.append(start.elapsed_time(end) / 1e3)

    device = resolve_device(device_name or detect_device())
    n_sel = int(len(sel_tiles))
    result = analyze(
        flops=gather_gemm_flops(m, n, k, tile_size=tile_size, n_selected=n_sel),
        bytes_moved=gather_gemm_bytes(m, n, k, tile_size=tile_size, n_selected=n_sel),
        times_s=times_s, device=device, dtype="fp4",
    )
    if verbose:
        dense_bytes = gather_gemm_bytes(m, n, k, tile_size=tile_size, n_selected=n_tiles(k, tile_size))
        gather_bytes = gather_gemm_bytes(m, n, k, tile_size=tile_size, n_selected=n_sel)
        print(f"\n[gss_gather_gemm] gather NVFP4 GEMM {m}x{n}x{k}, tile={tile_size}, "
              f"read {n_sel}/{n_tiles(k, tile_size)} tiles\n"
              f"  weight+act traffic: {gather_bytes:,} B vs dense {dense_bytes:,} B "
              f"({gather_bytes / dense_bytes:.3f}× = the bandwidth lever)\n")
        print(format_report(result))
    return result


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=1, help="rows of x (1 = token decode)")
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--tile", type=int, default=256, help="K-tile size (a unit / expert / channel group)")
    p.add_argument("--rho", type=float, default=0.10, help="read-set fraction of tiles to read")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    run_gss_gather_gemm(m=args.m, n=args.n, k=args.k, tile_size=args.tile, rho=args.rho,
                        iters=args.iters, warmup=args.warmup, device_name=args.device, seed=args.seed)
    return 0  # clean exit even on skip, so CI/orchestrator stay green pre-GPU


if __name__ == "__main__":
    sys.exit(main())
