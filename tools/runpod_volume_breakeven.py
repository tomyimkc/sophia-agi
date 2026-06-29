#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Break-even calculator: is a persistent RunPod network volume cheaper than re-downloading?

The hybrid-cluster cost plan flags that the launchers attach an EPHEMERAL ``volumeInGb`` disk
that dies when the pod is deleted (and the cost gate mandates always-delete). So under the
default setup, EVERY paid run pays the cold model-weight download as billed GPU minutes.

A persistent network volume (``--network-volume-id``) keeps the weights warm across pods, but
it costs a flat ``$/GB-month`` whether or not you run anything. It only pays off once you do
enough paid runs per month that the saved download time beats the idle rent.

This tool makes that GO/NO-GO explicit. It is pure-Python (no deps, CI-safe) and does NOT call
the RunPod API — it never spends. Numbers are estimates; pass your own with the flags.

    python tools/runpod_volume_breakeven.py --runs-per-month 12 --weights-gb 16 --gpu-hourly 1.69
"""
from __future__ import annotations

import argparse


def per_run_cold_cost(weights_gb: float, download_mbps: float, gpu_hourly: float) -> float:
    """$ of the cold weight download for ONE run = (download wall-clock, billed) x GPU $/hr.

    The download happens on the rented pod, so its minutes are billed at the GPU rate. A
    network volume removes this from every run after the first cache fill.
    """
    download_seconds = (weights_gb * 1024.0) / max(download_mbps, 1e-9)   # GB -> MB / (MB/s)
    return (download_seconds / 3600.0) * gpu_hourly


def analyze(*, runs_per_month: float, weights_gb: float, download_mbps: float,
            gpu_hourly: float, volume_gb: float, dollar_per_gb_month: float) -> dict:
    cold_one = per_run_cold_cost(weights_gb, download_mbps, gpu_hourly)
    monthly_download_cost = runs_per_month * cold_one          # what you pay WITHOUT a volume
    monthly_rent = volume_gb * dollar_per_gb_month             # flat cost of the volume, idle or not
    net_saving = monthly_download_cost - monthly_rent          # >0 => volume is cheaper
    breakeven_runs = (monthly_rent / cold_one) if cold_one > 0 else float("inf")
    return {
        "perRunColdDownloadUSD": round(cold_one, 4),
        "monthlyDownloadCostNoVolumeUSD": round(monthly_download_cost, 2),
        "monthlyVolumeRentUSD": round(monthly_rent, 2),
        "netSavingWithVolumeUSD": round(net_saving, 2),
        "breakevenRunsPerMonth": round(breakeven_runs, 1),
        "decision": "GO (volume cheaper)" if net_saving > 0 else "NO-GO (keep re-downloading)",
    }


def offline_invariants() -> "tuple[bool, dict]":
    """Self-checks (CI-safe): monotonicity + a known break-even point."""
    checks: dict[str, bool] = {}
    # More runs/month never makes the volume worse.
    a = analyze(runs_per_month=4, weights_gb=16, download_mbps=200, gpu_hourly=1.69,
                volume_gb=50, dollar_per_gb_month=0.07)
    b = analyze(runs_per_month=40, weights_gb=16, download_mbps=200, gpu_hourly=1.69,
                volume_gb=50, dollar_per_gb_month=0.07)
    checks["more_runs_better"] = b["netSavingWithVolumeUSD"] >= a["netSavingWithVolumeUSD"]
    # At exactly break-even runs, net saving ~ 0.
    cold = per_run_cold_cost(16, 200, 1.69)
    rent = 50 * 0.07
    be = rent / cold
    at_be = analyze(runs_per_month=be, weights_gb=16, download_mbps=200, gpu_hourly=1.69,
                    volume_gb=50, dollar_per_gb_month=0.07)
    checks["breakeven_zero"] = abs(at_be["netSavingWithVolumeUSD"]) < 0.01
    # A free-box ($0/hr) run has zero download cost, so a volume can never pay off.
    free = analyze(runs_per_month=1000, weights_gb=16, download_mbps=200, gpu_hourly=0.0,
                   volume_gb=50, dollar_per_gb_month=0.07)
    checks["free_gpu_never_go"] = free["decision"].startswith("NO-GO")
    return all(checks.values()), {"checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-per-month", type=float, default=8, help="paid RunPod runs/month that reuse these weights")
    ap.add_argument("--weights-gb", type=float, default=16, help="cold model+cache download size in GB")
    ap.add_argument("--download-mbps", type=float, default=200, help="effective download speed MB/s on the pod")
    ap.add_argument("--gpu-hourly", type=float, default=1.69, help="GPU $/hr (download minutes are billed at this)")
    ap.add_argument("--volume-gb", type=float, default=50, help="network volume size in GB")
    ap.add_argument("--dollar-per-gb-month", type=float, default=0.07, help="RunPod network-volume storage $/GB/month")
    ap.add_argument("--self-test", action="store_true", help="run offline invariants and exit")
    args = ap.parse_args()

    if args.self_test:
        ok, detail = offline_invariants()
        print("breakeven offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1

    r = analyze(runs_per_month=args.runs_per_month, weights_gb=args.weights_gb,
                download_mbps=args.download_mbps, gpu_hourly=args.gpu_hourly,
                volume_gb=args.volume_gb, dollar_per_gb_month=args.dollar_per_gb_month)
    print("RunPod network-volume break-even")
    print(f"  inputs: {args.runs_per_month:g} runs/mo, {args.weights_gb:g}GB cold @ {args.download_mbps:g}MB/s, "
          f"GPU ${args.gpu_hourly:g}/hr, volume {args.volume_gb:g}GB @ ${args.dollar_per_gb_month:g}/GB-mo")
    print(f"  per-run cold download .......... ${r['perRunColdDownloadUSD']}")
    print(f"  monthly download cost (no vol) . ${r['monthlyDownloadCostNoVolumeUSD']}")
    print(f"  monthly volume rent ............ ${r['monthlyVolumeRentUSD']}")
    print(f"  net saving with volume ......... ${r['netSavingWithVolumeUSD']}")
    print(f"  break-even at .................. {r['breakevenRunsPerMonth']} runs/month")
    print(f"  >>> {r['decision']}")
    print("  note: free Spark/Mac runs cost $0 -> a volume only ever helps the PAID burst lane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
