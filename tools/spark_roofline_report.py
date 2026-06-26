#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Spark roofline report harness — run the kernels on a DGX Spark, report % of *its* ceiling.

This is the local-iteration sibling of ``tools/runpod_kernels.py``: it runs the repo's
GPU kernels (the BF16 tiled GEMM and the fused NVFP4 dequant-GEMM) on a DGX Spark GB10
and emits a roofline report measured against the Spark's own datasheet ceiling
(273 GB/s unified LPDDR5x; ~500 TFLOP/s dense FP4) — never an "Nx vs my laptop" number.

**Provenance boundary (load-bearing, same as the Spark lane).** Every report this writes
is annotated ``sparkIteration: true, registeredResult: false``. A Spark number is for
fast, free local iteration; it is *not* a registered result (aarch64 numerics differ from
the x86 RunPod source-of-record). See docs/11-Platform/Spark-Local-GPU-Lane.md and
docs/11-Platform/DGX-Spark-Maximization.md.

Offline by default: ``--dry-run`` (the default) prints the plan + the Spark device profile
and writes nothing, so CI and a Spark-less dev box stay green. With ``--run`` on an actual
Spark (torch+CUDA+triton) it times the kernels and writes a JSON report under
``kernels/reports/`` (git-ignored).

    python tools/spark_roofline_report.py                 # dry-run: print the plan, no GPU
    python tools/spark_roofline_report.py --run           # on a real Spark: time + write report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.bench.roofline import DEVICE_SPECS, resolve_device  # noqa: E402

SPARK_DEVICE = "NVIDIA DGX Spark GB10"
REPORTS_DIR = ROOT / "kernels" / "reports"


def _safe_stamp(stamp: str) -> str:
    """Reduce ``stamp`` to a filename-safe token so it cannot escape REPORTS_DIR.

    The stamp is interpolated into the report filename; a raw value could contain
    path separators (``../../x``) and write outside ``kernels/reports/``. Keep only
    ``[A-Za-z0-9._-]`` and reject an empty result fail-closed.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", str(stamp)).strip("-.")
    if not safe:
        raise ValueError(f"--stamp {stamp!r} has no filename-safe characters")
    return safe

# The kernels this harness runs, each a decode-or-compute-shaped problem on the Spark.
PLAN = [
    {"kernel": "bf16_gemm", "shape": {"m": 4096, "n": 4096, "k": 4096},
     "what": "Triton tiled BF16 GEMM (compute-leaning; the M1 reference kernel)"},
    {"kernel": "nvfp4_gemm", "shape": {"m": 1, "n": 8192, "k": 8192},
     "what": "fused NVFP4 dequant-GEMM, decode-shaped (memory-bound; the Spark-native path)"},
]


def build_report(results: list[dict], *, device_name: str = SPARK_DEVICE,
                 stamp: str = "unstamped") -> dict:
    """Assemble the annotated report dict. Pure (no clock/GPU) so it is unit-testable.

    ``results`` is a list of ``{"kernel","shape","roofline": <RooflineResult.to_dict()|None>}``.
    The provenance annotations are non-negotiable and always present.
    """
    device = resolve_device(device_name)
    return {
        "harness": "spark_roofline_report",
        "device": device.name,
        "deviceNote": device.note,
        "stamp": stamp,
        # Provenance boundary — see the Spark lane doc. These MUST be present and fixed.
        "sparkIteration": True,
        "registeredResult": False,
        "provenanceNote": (
            "Spark (aarch64, bf16/NVFP4) numerics differ from the x86 RunPod source-of-record; "
            "this is a free local iteration, not a registered result."
        ),
        "ceiling": {
            "bandwidth_gbytes_s": device.hbm_gbytes_s,
            "fp4_tflops_dense": device.fp4_tensor_tflops,
            "bf16_tflops_dense": device.fp16_tensor_tflops,
        },
        "results": results,
    }


def _print_plan(device_name: str) -> None:
    device = resolve_device(device_name)
    print(f"Spark roofline plan — device: {device.name}")
    print(f"  ceiling: {device.hbm_gbytes_s:.0f} GB/s unified  |  "
          f"{device.fp4_tensor_tflops:.0f} TFLOP/s dense FP4  |  "
          f"{device.fp16_tensor_tflops:.0f} TFLOP/s dense BF16")
    for i, p in enumerate(PLAN, 1):
        s = p["shape"]
        print(f"  [{i}] {p['kernel']:11s} {s['m']}x{s['n']}x{s['k']}  — {p['what']}")
    print("\nProvenance: every written report is sparkIteration=true, registeredResult=false.")
    print("Dry-run: nothing was executed or written. Re-run with --run on a real Spark.")


def _run_live(device_name: str, iters: int, stamp: str) -> int:
    from kernels.src.nvfp4_gemm import run_nvfp4_gemm
    from kernels.src.run_kernel import run_gemm

    results: list[dict] = []
    for p in PLAN:
        s = p["shape"]
        if p["kernel"] == "bf16_gemm":
            r = run_gemm(m=s["m"], n=s["n"], k=s["k"], iters=iters, device_name=device_name)
        else:
            r = run_nvfp4_gemm(m=s["m"], n=s["n"], k=s["k"], iters=iters, device_name=device_name)
        results.append({"kernel": p["kernel"], "shape": s,
                        "roofline": r.to_dict() if r is not None else None})

    stamp = _safe_stamp(stamp)
    report = build_report(results, device_name=device_name, stamp=stamp)
    ran_any = any(x["roofline"] is not None for x in results)
    if not ran_any:
        print("[spark_roofline_report] no kernel ran (no torch+CUDA+triton). "
              "Wrote nothing; this is not a Spark.")
        return 0
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"spark-roofline-{stamp}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[spark_roofline_report] wrote {out} "
          f"(sparkIteration=true, registeredResult=false).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="actually run on a Spark (default: dry-run)")
    p.add_argument("--device", default=SPARK_DEVICE, help="device profile name (must be in DEVICE_SPECS)")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--stamp", default="manual", help="report filename stamp (no clock used here)")
    args = p.parse_args(argv)

    if args.device not in DEVICE_SPECS and not any(
        args.device.lower() in k.lower() for k in DEVICE_SPECS
    ):
        print(f"unknown device {args.device!r}; known: {', '.join(sorted(DEVICE_SPECS))}")
        return 2

    if not args.run:
        _print_plan(args.device)
        return 0
    return _run_live(args.device, args.iters, args.stamp)


if __name__ == "__main__":
    sys.exit(main())
