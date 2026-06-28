#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Benchmark FlashAttention reference vs naive: correctness + score-memory.

OFFLINE (CI-safe): for growing sequence length N, confirm the flash reference
agrees with naive attention and report the peak score-tile memory the flash path
holds vs naive's full N² matrix — the quantity FlashAttention reduces. This is a
*memory-traffic* argument, which is hardware-independent and the real reason the
kernel matters; raw numpy wall-time is not representative of a fused GPU kernel,
so it's reported only as a sanity signal, never as a speedup claim.

LIVE (gated): if Triton + CUDA are present, also run the fused kernel and check
it matches the reference.

    python kernels/bench.py
    python kernels/bench.py --seqlens 128,512,2048
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import flash_attention as fa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqlens", default="64,128,256,512")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()

    try:
        import numpy as np
    except Exception:
        print("numpy required for the benchmark")
        return 1

    seqlens = [int(x) for x in args.seqlens.split(",")]
    rng = np.random.default_rng(0)
    print(f"{'N':>6} {'match':>6} {'flash_tile':>11} {'full_NxN':>10} "
          f"{'mem_reduction':>13} {'naive_ms':>9} {'flash_ms':>9}")
    all_ok = True
    for N in seqlens:
        Q = rng.standard_normal((N, args.dim))
        K = rng.standard_normal((N, args.dim))
        V = rng.standard_normal((N, args.dim))
        t0 = time.perf_counter()
        ref = fa.naive_attention(Q, K, V, causal=args.causal)
        t1 = time.perf_counter()
        stats: dict = {}
        out = fa.flash_attention_numpy(
            Q, K, V, block_q=args.block, block_k=args.block,
            causal=args.causal, stats=stats,
        )
        t2 = time.perf_counter()
        match = bool(np.allclose(ref, out, atol=1e-9, rtol=1e-9))
        all_ok &= match
        red = stats["full_matrix"] / max(1, stats["max_score_tile"])
        print(f"{N:>6} {str(match):>6} {stats['max_score_tile']:>11} "
              f"{stats['full_matrix']:>10} {red:>12.1f}x "
              f"{(t1-t0)*1e3:>8.2f} {(t2-t1)*1e3:>8.2f}")

    if fa.triton_available():
        print("\nTriton+CUDA detected — running fused kernel check:")
        N = seqlens[-1]
        Q = rng.standard_normal((N, args.dim)).astype("float32")
        ref = fa.naive_attention(Q, Q, Q, causal=args.causal)
        out = fa.flash_attention_triton(Q, Q, Q, causal=args.causal)
        print(f"  triton matches reference: {np.allclose(ref, out, atol=1e-2)}")
    else:
        print("\n(Triton/CUDA unavailable — fused kernel path skipped; "
              "numpy reference is the CI-tested artifact.)")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
