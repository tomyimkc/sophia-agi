#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""On-pod NCCL all-reduce micro-benchmark — measure real intra-node GPU interconnect bandwidth.

Run under torchrun with one process per GPU; each rank participates in a ring all-reduce
over a sweep of message sizes. Rank 0 records, for each size, the median step time and the
NCCL 'bus bandwidth' (busbw = algbw · 2(N-1)/N) — the same convention as nvidia/nccl-tests,
so the numbers are comparable. The report feeds tools/calibrate_network_tax.py, which
replaces the simulator's MODELED NVLink tier with this MEASURED bandwidth.

This is the on-pod payload only (needs CUDA + torch); it is launched by
tools/runpod_nccl_bench.py. torch is imported lazily so --dry-run / --emit-cmd and the
unit tests run with pure stdlib.

    # what runs on the pod (N = #GPUs):
    torchrun --standalone --nproc_per_node=N tools/bench_nccl_allreduce.py --run --out report.json
    # locally, no GPU:
    python tools/bench_nccl_allreduce.py --dry-run --gpus 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_SIZES_MB = (1, 4, 16, 64, 256, 512)
DEFAULT_ITERS = 20
DEFAULT_WARMUP = 5


def build_torchrun_cmd(nproc: int, out: str, *, sizes_mb=DEFAULT_SIZES_MB,
                       iters: int = DEFAULT_ITERS, script: str = "tools/bench_nccl_allreduce.py") -> str:
    """The torchrun invocation that runs this benchmark across `nproc` GPUs (pure string)."""
    sizes = ",".join(str(s) for s in sizes_mb)
    return (f"torchrun --standalone --nproc_per_node={nproc} {script} "
            f"--run --out {out} --sizes-mb {sizes} --iters {iters}")


def _worker_main(args: argparse.Namespace) -> int:
    """Runs under torchrun: each rank joins the all-reduce; rank 0 writes the report."""
    import torch  # noqa: E402 — lazy: only available on the GPU pod
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        print("CUDA not available — this benchmark must run on a GPU pod", file=sys.stderr)
        return 2
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")

    sizes_bytes = [int(mb) * 1024 * 1024 for mb in args.sizes_mb.split(",")]
    results = []
    for nbytes_ in sizes_bytes:
        n_elems = max(1, nbytes_ // 4)  # float32
        buf = torch.ones(n_elems, dtype=torch.float32, device=f"cuda:{local}")
        for _ in range(args.warmup):
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(args.iters):
            start.record()
            dist.all_reduce(buf)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
        times_ms.sort()
        median_s = times_ms[len(times_ms) // 2] / 1e3
        algbw = (nbytes_ / median_s) / 1e9 if median_s else 0.0
        busbw = algbw * 2.0 * (world - 1) / world if world > 1 else 0.0
        if rank == 0:
            results.append({
                "size_bytes": nbytes_,
                "time_s": round(median_s, 8),
                "algbw_gbps": round(algbw, 3),
                "busbw_gbps": round(busbw, 3),
            })
    if rank == 0:
        report = {
            "schema": "sophia.nccl_allreduce.v1",
            "scope": "MEASURED on one RunPod multi-GPU pod (intra-node NVLink/NVSwitch).",
            "n_ranks": world,
            "gpu_name": torch.cuda.get_device_name(local),
            "iters": args.iters,
            "results": results,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="store_true", help="execute the benchmark (under torchrun)")
    p.add_argument("--dry-run", action="store_true", help="print the plan + torchrun command")
    p.add_argument("--emit-cmd", action="store_true", help="print only the torchrun command")
    p.add_argument("--gpus", type=int, default=8, help="GPU count for dry-run/emit-cmd")
    p.add_argument("--sizes-mb", default=",".join(str(s) for s in DEFAULT_SIZES_MB))
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--out", default="agi-proof/benchmark-results/cluster/nccl-allreduce.public-report.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run:
        return _worker_main(args)
    cmd = build_torchrun_cmd(args.gpus, args.out,
                             sizes_mb=[int(x) for x in args.sizes_mb.split(",")], iters=args.iters)
    if args.emit_cmd:
        print(cmd)
        return 0
    # default: dry-run plan
    print("NCCL all-reduce benchmark plan (DRY-RUN — no GPU touched)")
    print(f"  gpus      : {args.gpus}")
    print(f"  sizes (MB): {args.sizes_mb}")
    print(f"  iters     : {args.iters} (+{args.warmup} warmup)")
    print(f"  out       : {args.out}")
    print(f"  command   : {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
