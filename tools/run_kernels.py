#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Kernels M0 — reference correctness + roofline analysis (offline, no GPU).

Validates the pure-stdlib reference kernels (softmax / RMSNorm / SwiGLU) and shows,
via the existing roofline harness, that all three are memory-bound on the target
device — the offline justification for fusing them. The actual Triton kernels
(`kernels/triton/`) run + benchmark on the DGX Spark / RunPod; correctness there is
defined as matching `kernels.reference` within tolerance.

    python tools/run_kernels.py --mode mock --device "NVIDIA DGX Spark GB10"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import reference as ref  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "kernels" / "kernels.public-report.json"


def _offline_invariants(device: str) -> "tuple[bool, dict]":
    sm = ref.softmax([1.0, 2.0, 3.0, 4.0])
    rn = ref.rmsnorm([3.0, 4.0], [1.0, 1.0])
    sg = ref.swiglu([0.0, 1.0], [2.0, 3.0])
    rows, d = 4096, 4096
    regimes = {op: ref.classify(op, rows, d, device) for op in ("softmax", "rmsnorm", "swiglu")}

    checks = {
        "softmaxSumsToOne": abs(sum(sm) - 1.0) < 1e-12,
        "softmaxMonotone": sm[0] < sm[1] < sm[2] < sm[3],
        "rmsnormUnitScale": abs((rn[0] ** 2 + rn[1] ** 2) / 2 - 1.0) < 1e-6,
        "swigluMatchesSiluUp": abs(sg[0] - ref.silu(0.0) * 2.0) < 1e-12
                               and abs(sg[1] - ref.silu(1.0) * 3.0) < 1e-12,
        "allMemoryBound": all(r["regime"] == "memory-bound" for r in regimes.values()),
        "ridgeAboveIntensity": all(r["intensity"] < r["ridgePoint"] for r in regimes.values()),
    }
    return all(checks.values()), {
        "device": device,
        "regimes": regimes,
        "checks": checks,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["mock"], default="mock")
    ap.add_argument("--device", default="NVIDIA DGX Spark GB10")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    ok, detail = _offline_invariants(args.device)
    report = {
        "benchmark": "kernels-reference-roofline",
        "mode": "mock-offline",
        "visibility": "public-aggregate",
        "claimStatus": "Reference numerics + roofline accounting only. Measured kernel "
                       "speedups (% of HBM SOL, ncu-attributed) require the Triton kernels on "
                       "a CUDA device; correctness there = matching kernels.reference.",
        "candidateOnly": True,
        "canClaimAGI": False,
        **detail,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if ok:
        print(f"KERNELS OFFLINE CORE VERIFIED ✓  (softmax/rmsnorm/swiglu memory-bound on {args.device})")
        return 0
    print(f"KERNELS OFFLINE CORE FAILED ✗  failing: "
          f"{[k for k, v in detail['checks'].items() if not v]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
