#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Roofline harness — measure a kernel against the hardware's physical limit.

This is the *gate* for the HPC operator track (docs/06-Roadmap/HPC-Operator-Compiler-Roadmap.md).
It exists so that the very first kernel we write is born already measured against the
theoretical ceiling — never reported as "Nx faster than a naive baseline".

What it computes, for a kernel that does ``flops`` floating-point ops while moving
``bytes_moved`` bytes to/from HBM in wall-time ``t``:

  achieved        = flops / t                      (FLOP/s)
  bandwidth       = bytes_moved / t                (bytes/s)
  intensity  I    = flops / bytes_moved            (FLOP/byte, arithmetic intensity)
  ceiling  P*     = min(peak_compute, I * peak_bw) (the roofline — attainable peak)
  % of roofline   = achieved / P*                  (distance to the *physical* limit)
  regime          = memory-bound iff I < P/B (left of the ridge point), else compute-bound

The roofline model (Williams, Waterman & Patterson, CACM 2009) says a kernel can never
exceed ``min(peak_compute, I * peak_bandwidth)``. "% of roofline" is therefore the honest
headline number this team cares about — and it is bounded by construction at 100%.

Peak numbers in ``DEVICE_SPECS`` are *vendor datasheet theoretical* peaks (dense, no
sparsity), not measured. They are the denominator, not a claim. Treat anything above ~95%
of roofline as suspicious (re-check the FLOP/byte accounting) rather than a triumph.

Offline by default — the math needs no GPU. ``torch`` is used only to *detect* the device
and (optionally) *time* a real kernel; without it, pass a known device name and timings.

    python kernels/bench/roofline.py --demo                 # synthetic GEMM, offline
    python kernels/bench/roofline.py --demo --device "NVIDIA H100 80GB HBM3"
    python kernels/bench/roofline.py --self-test            # exercise the math
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field


# --------------------------------------------------------------------------------------
# Device peaks — VENDOR DATASHEET THEORETICAL (dense, no 2:4 sparsity). The denominator.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceSpec:
    name: str
    fp16_tensor_tflops: float  # BF16/FP16 tensor-core peak, dense (TFLOP/s)
    fp32_tflops: float         # FP32 non-tensor peak (TFLOP/s)
    hbm_gbytes_s: float        # HBM/unified bandwidth (GB/s, 1e9 bytes)
    note: str = ""
    fp4_tensor_tflops: float = 0.0  # NVFP4/FP4 tensor-core peak, DENSE (TFLOP/s); 0 == unsupported

    def peak_flops(self, dtype: str) -> float:
        """Peak FLOP/s for a dtype tier (fp4 -> FP4 tensor, fp16/bf16 -> tensor, else fp32)."""
        d = dtype.lower()
        if d in {"fp4", "nvfp4", "mxfp4", "e2m1"}:
            if self.fp4_tensor_tflops <= 0:
                raise ValueError(
                    f"{self.name} has no FP4 tensor peak in DEVICE_SPECS "
                    "(pre-Blackwell, or not yet entered) — cannot roofline FP4 here"
                )
            return self.fp4_tensor_tflops * 1e12
        tier = self.fp16_tensor_tflops if d in {"fp16", "bf16", "half"} else self.fp32_tflops
        return tier * 1e12

    def peak_bw(self) -> float:
        return self.hbm_gbytes_s * 1e9


# Datasheet peaks. Sources: NVIDIA A100 / H100 datasheets (dense tensor, no sparsity).
DEVICE_SPECS: dict[str, DeviceSpec] = {
    "NVIDIA A100 80GB PCIe": DeviceSpec("NVIDIA A100 80GB PCIe", 312.0, 19.5, 1935.0, "Ampere, HBM2e"),
    "NVIDIA A100-SXM4-80GB": DeviceSpec("NVIDIA A100-SXM4-80GB", 312.0, 19.5, 2039.0, "Ampere, HBM2e"),
    "NVIDIA H100 PCIe": DeviceSpec("NVIDIA H100 PCIe", 756.0, 51.0, 2000.0, "Hopper, HBM2e/HBM3"),
    "NVIDIA H100 80GB HBM3": DeviceSpec("NVIDIA H100 80GB HBM3", 989.5, 67.0, 3350.0, "Hopper SXM, HBM3"),
    "NVIDIA L40S": DeviceSpec("NVIDIA L40S", 362.0, 91.6, 864.0, "Ada, GDDR6"),
    "NVIDIA GeForce RTX 4090": DeviceSpec("NVIDIA GeForce RTX 4090", 165.2, 82.6, 1008.0, "Ada, GDDR6X"),
    # Grace Blackwell GB10 (DGX Spark), aarch64. Bandwidth is LPDDR5x UNIFIED memory (273 GB/s,
    # confirmed by NVIDIA's DGX Spark hardware overview), NOT HBM — so the roofline ridge point
    # sits FAR to the right: even FP4 GEMM needs intensity > ~1830 FLOP/B to leave the memory wall,
    # so token decode is memory-bound here and bytes-per-weight (4-bit) is the dominant lever.
    # NVIDIA markets GB10 at ~1 PFLOP *sparse* FP4; the FP4 dense peak below is ~half that (sparse 2:4
    # doubles the headline). BF16/FP32 dense peaks are estimates — VERIFY vs datasheet before any
    # headline derived from these denominators.
    "NVIDIA DGX Spark GB10": DeviceSpec(
        "NVIDIA DGX Spark GB10", 125.0, 31.0, 273.0,
        "Grace Blackwell GB10 (aarch64); 128GB LPDDR5x UNIFIED 273 GB/s (NOT HBM); FP4 ~500 TFLOP/s "
        "dense (~1 PFLOP sparse); BF16/FP32 peaks APPROXIMATE — verify vs datasheet",
        fp4_tensor_tflops=500.0),
}


def resolve_device(name: str | None) -> DeviceSpec:
    """Return a DeviceSpec by exact or fuzzy name; raise if unknown (never guess peaks)."""
    if not name:
        raise ValueError("no device given and none detected; pass --device <name>")
    if name in DEVICE_SPECS:
        return DEVICE_SPECS[name]
    low = name.lower()
    for key, spec in DEVICE_SPECS.items():
        if low in key.lower() or key.lower() in low:
            return spec
    known = ", ".join(sorted(DEVICE_SPECS))
    raise ValueError(f"unknown device {name!r}; known: {known} (add its datasheet peaks to DEVICE_SPECS)")


def detect_device() -> str | None:
    """Best-effort device name via torch; None offline (so the math still runs)."""
    try:
        import torch  # noqa: PLC0415 — optional, GPU-only path

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------------------
# FLOP / byte accounting helpers — kept explicit so the denominator is auditable.
# --------------------------------------------------------------------------------------
def gemm_flops(m: int, n: int, k: int) -> int:
    """C[m,n] = A[m,k] @ B[k,n]: 2*m*n*k (one multiply + one add per MAC)."""
    return 2 * m * n * k


def gemm_bytes(m: int, n: int, k: int, dtype_bytes: int = 2) -> int:
    """Lower bound on HBM traffic: read A, read B, write C, each touched once."""
    return dtype_bytes * (m * k + k * n + m * n)


def attention_flops(batch: int, heads: int, seq: int, head_dim: int, causal: bool = False) -> int:
    """FlashAttention-style: QK^T and (softmax·V), each 2*S*S*d MACs per head.

    Causal masking roughly halves the score work; we report the dense upper bound and
    scale by 0.5 when causal (an approximation — document it where used)."""
    per_head = 2 * (2 * seq * seq * head_dim)
    total = batch * heads * per_head
    return total // 2 if causal else total


# --------------------------------------------------------------------------------------
# The roofline result.
# --------------------------------------------------------------------------------------
@dataclass
class RooflineResult:
    device: str
    dtype: str
    flops: int
    bytes_moved: int
    runs: int
    time_s_median: float
    time_s_min: float
    time_s_stdev: float
    achieved_tflops: float
    achieved_gbytes_s: float
    arithmetic_intensity: float          # FLOP/byte
    ridge_point: float                   # FLOP/byte where compute & bw ceilings cross
    regime: str                          # "memory-bound" | "compute-bound"
    peak_tflops: float                   # the relevant compute ceiling (datasheet)
    peak_gbytes_s: float                 # HBM ceiling (datasheet)
    roofline_ceiling_tflops: float       # min(compute, I*bw) — attainable peak
    pct_of_roofline: float               # achieved / ceiling  (THE headline, <=~100%)
    pct_of_compute_peak: float           # achieved / datasheet compute peak
    pct_of_bandwidth_peak: float         # achieved bw / datasheet bw
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(
    *,
    flops: int,
    bytes_moved: int,
    times_s: list[float],
    device: DeviceSpec,
    dtype: str = "bf16",
) -> RooflineResult:
    """Build a RooflineResult from raw FLOP/byte counts and a list of timings.

    ``times_s`` should hold >=3 trials so dispersion is real. We use the *median* for the
    headline (robust to a warmup/outlier) and surface min + stdev for honesty.
    """
    if flops <= 0 or bytes_moved <= 0:
        raise ValueError("flops and bytes_moved must be positive")
    times = [t for t in times_s if t and t > 0]
    if not times:
        raise ValueError("need at least one positive timing")

    warnings: list[str] = []
    if len(times) < 3:
        warnings.append(f"only {len(times)} timing(s); the no-overclaim gate wants >=3 runs")

    t_med = statistics.median(times)
    t_min = min(times)
    t_std = statistics.stdev(times) if len(times) > 1 else 0.0

    achieved = flops / t_med                       # FLOP/s
    bw = bytes_moved / t_med                        # bytes/s
    intensity = flops / bytes_moved                 # FLOP/byte

    peak_compute = device.peak_flops(dtype)         # FLOP/s
    peak_bw = device.peak_bw()                      # bytes/s
    ridge = peak_compute / peak_bw                  # FLOP/byte
    ceiling = min(peak_compute, intensity * peak_bw)
    regime = "memory-bound" if intensity < ridge else "compute-bound"

    pct_roof = achieved / ceiling if ceiling else 0.0
    if pct_roof > 1.0 + 1e-6:
        warnings.append(
            f"achieved {pct_roof:.1%} of the roofline (>100%) — FLOP/byte accounting is "
            "almost certainly wrong, or the device peaks are too low. Re-check before reporting."
        )

    return RooflineResult(
        device=device.name,
        dtype=dtype,
        flops=int(flops),
        bytes_moved=int(bytes_moved),
        runs=len(times),
        time_s_median=t_med,
        time_s_min=t_min,
        time_s_stdev=t_std,
        achieved_tflops=achieved / 1e12,
        achieved_gbytes_s=bw / 1e9,
        arithmetic_intensity=intensity,
        ridge_point=ridge,
        regime=regime,
        peak_tflops=peak_compute / 1e12,
        peak_gbytes_s=peak_bw / 1e9,
        roofline_ceiling_tflops=ceiling / 1e12,
        pct_of_roofline=pct_roof,
        pct_of_compute_peak=achieved / peak_compute if peak_compute else 0.0,
        pct_of_bandwidth_peak=bw / peak_bw if peak_bw else 0.0,
        warnings=warnings,
    )


def format_report(r: RooflineResult) -> str:
    """Human-readable, RESULTS.md-flavored block. The headline is % of roofline."""
    lines = [
        f"device           : {r.device}  ({r.dtype})",
        f"problem          : {r.flops:,} FLOP, {r.bytes_moved:,} B HBM, "
        f"arithmetic intensity {r.arithmetic_intensity:.2f} FLOP/B",
        f"regime           : {r.regime}  (ridge point {r.ridge_point:.2f} FLOP/B)",
        f"time             : median {r.time_s_median * 1e3:.4f} ms  "
        f"(min {r.time_s_min * 1e3:.4f} ms, stdev {r.time_s_stdev * 1e3:.4f} ms, {r.runs} runs)",
        f"achieved         : {r.achieved_tflops:.2f} TFLOP/s, {r.achieved_gbytes_s:.1f} GB/s",
        f"datasheet peak   : {r.peak_tflops:.1f} TFLOP/s compute, {r.peak_gbytes_s:.0f} GB/s HBM",
        f"roofline ceiling : {r.roofline_ceiling_tflops:.2f} TFLOP/s (attainable @ this intensity)",
        "",
        f">>> % OF ROOFLINE : {r.pct_of_roofline:6.1%}   <-- distance to the physical limit",
        f"    % compute peak: {r.pct_of_compute_peak:6.1%}",
        f"    % HBM peak    : {r.pct_of_bandwidth_peak:6.1%}",
    ]
    for w in r.warnings:
        lines.append(f"!! WARNING: {w}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _demo(args: argparse.Namespace) -> int:
    device = resolve_device(args.device or detect_device() or "NVIDIA A100-SXM4-80GB")
    m, n, k = args.m, args.n, args.k
    flops = gemm_flops(m, n, k)
    bytes_moved = gemm_bytes(m, n, k, dtype_bytes=2)
    # A deliberately synthetic timing: pretend we hit `frac` of the datasheet compute peak.
    # This exercises the math offline; it is NOT a measurement.
    frac = args.synthetic_frac
    t = flops / (device.peak_flops(args.dtype) * frac)
    times = [t * j for j in (1.00, 1.02, 0.99)]  # fake 3-run jitter
    r = analyze(flops=flops, bytes_moved=bytes_moved, times_s=times, device=device, dtype=args.dtype)
    print(f"[demo] synthetic GEMM {m}x{n}x{k} @ {frac:.0%} of datasheet compute peak "
          f"(NOT a measurement)\n")
    if args.json:
        print(json.dumps(r.to_dict(), indent=2))
    else:
        print(format_report(r))
    return 0


def _self_test(_args: argparse.Namespace) -> int:
    """Exercise the math against hand-checkable invariants."""
    dev = DEVICE_SPECS["NVIDIA A100-SXM4-80GB"]
    # Compute-bound: a square GEMM has very high intensity -> regime compute-bound.
    m = n = k = 4096
    f, b = gemm_flops(m, n, k), gemm_bytes(m, n, k, 2)
    # Time that exactly hits 50% of compute peak.
    t = f / (dev.peak_flops("bf16") * 0.5)
    r = analyze(flops=f, bytes_moved=b, times_s=[t, t, t], device=dev, dtype="bf16")
    assert r.regime == "compute-bound", r.regime
    assert abs(r.pct_of_compute_peak - 0.5) < 1e-3, r.pct_of_compute_peak
    assert r.pct_of_roofline <= 1.0 + 1e-6, r.pct_of_roofline
    # Memory-bound: a tiny-k, fat GEMM (low intensity) should be memory-bound.
    f2, b2 = gemm_flops(4096, 4096, 1), gemm_bytes(4096, 4096, 1, 2)
    t2 = b2 / (dev.peak_bw() * 0.8)  # hit 80% of HBM
    r2 = analyze(flops=f2, bytes_moved=b2, times_s=[t2, t2, t2], device=dev, dtype="bf16")
    assert r2.regime == "memory-bound", r2.regime
    assert abs(r2.pct_of_bandwidth_peak - 0.8) < 1e-2, r2.pct_of_bandwidth_peak
    # Over-100% guard fires.
    r3 = analyze(flops=f, bytes_moved=b, times_s=[t / 4], device=dev, dtype="bf16")
    assert any("roofline" in w for w in r3.warnings), r3.warnings
    print("self-test OK:",
          f"compute-bound={r.pct_of_compute_peak:.1%},",
          f"memory-bound={r2.pct_of_bandwidth_peak:.1%} HBM,",
          "over-100% guard fired.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="GPU name (see DEVICE_SPECS); autodetected if torch+CUDA")
    p.add_argument("--dtype", default="bf16", help="bf16/fp16 -> tensor peak; else fp32 peak")
    p.add_argument("--demo", action="store_true", help="run the offline synthetic GEMM demo")
    p.add_argument("--self-test", action="store_true", help="exercise the roofline math and exit")
    p.add_argument("--json", action="store_true", help="emit the result as JSON")
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--synthetic-frac", type=float, default=0.45,
                   help="demo only: pretend we hit this fraction of datasheet compute peak")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test(args)
    if args.demo:
        return _demo(args)
    # Default: list known devices so the harness is self-describing.
    print("Roofline harness. Known devices (datasheet theoretical peaks):\n")
    for spec in DEVICE_SPECS.values():
        print(f"  {spec.name:28s}  {spec.fp16_tensor_tflops:7.1f} TFLOP/s tensor  "
              f"{spec.hbm_gbytes_s:6.0f} GB/s HBM   ({spec.note})")
    print("\nRun with --demo (offline synthetic) or --self-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
