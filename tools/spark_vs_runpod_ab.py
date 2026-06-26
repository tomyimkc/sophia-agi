#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Quantify the aarch64/GB10 (DGX Spark) vs x86 (RunPod) divergence on the SAME config.

Per REPLICATION.md, headline numbers stay on x86 RunPod; the Spark is the iteration tier.
That rule should be DATA-backed, not asserted. This harness compares two held-out
adapter-eval reports (tools/eval_rlvr_adapter.py output) — one produced on the Spark
(`--local`), one on a RunPod dispatch (x86) — for the SAME task/model/seed, and records
the divergence. A small, stable divergence justifies treating the Spark as a faithful
iteration proxy; a large one quantifies why its numbers can't be cited as the result.

Inputs are the two eval reports (task-aware: code/math use passAt1, provenance uses
meanReward — same extraction as tools/ingest_rlvr_eval.map_report). Mock mode synthesizes
two reports so the comparison logic is CI-verifiable without any GPU.

    python tools/spark_vs_runpod_ab.py \
        --local-report agi-proof/.../spark.adapter-eval.json \
        --runpod-report agi-proof/.../runpod.adapter-eval.json \
        --out agi-proof/.../spark-vs-runpod-ab.json
    python tools/spark_vs_runpod_ab.py --mock --out /tmp/ab.json   # CI logic check
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _capability(report: dict) -> tuple[str, float, float]:
    """Return (metric_name, base, adapter) for an eval report (task-aware)."""
    task = report.get("task") or "provenance"
    metric = "passAt1" if task in ("math", "code") else "meanReward"
    base = report.get("base", {}).get(metric, report.get("baseScore", {}).get(metric))
    adapter = report.get("adapterScore", {}).get(metric, report.get("adapter", {}).get(metric))
    if base is None or adapter is None:
        raise ValueError(f"report missing base/adapterScore.{metric} (task={task})")
    return metric, float(base), float(adapter)


def compare(local: dict, runpod: dict) -> dict[str, Any]:
    """Compare two eval reports (same task/config, different hardware tiers)."""
    metric, l_base, l_adapter = _capability(local)
    r_metric, r_base, r_adapter = _capability(runpod)
    if metric != r_metric:
        raise ValueError(f"task mismatch: local={metric} runpod={r_metric}")
    l_delta = round(l_adapter - l_base, 4)
    r_delta = round(r_adapter - r_base, 4)
    return {
        "schema": "sophia.spark_vs_runpod_ab.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "metric": metric,
        "task": local.get("task", "provenance"),
        "model": local.get("model"),
        "spark": {"base": l_base, "adapter": l_adapter, "delta": l_delta,
                  "source": "DGX Spark (aarch64/GB10, iteration tier)"},
        "runpod": {"base": r_base, "adapter": r_adapter, "delta": r_delta,
                   "source": "RunPod (x86, headline tier)"},
        "divergence": {
            "adapterAbsolute": round(abs(l_adapter - r_adapter), 4),
            "deltaAbsolute": round(abs(l_delta - r_delta), 4),
            "baseAbsolute": round(abs(l_base - r_base), 4),
        },
        "interpretation": (
            f"Adapter {metric} divergence Spark-vs-RunPod = {abs(l_adapter - r_adapter):.4f}. "
            "Small + stable divergence => the Spark is a faithful ITERATION proxy (iterate locally, "
            "register on x86). Large divergence => the Spark number must NOT be cited as the result. "
            "Either way, per REPLICATION.md the headline stays on x86 RunPod. canClaimAGI=false."
        ),
    }


def _mock_reports() -> tuple[dict, dict]:
    """Two synthetic code-task eval reports: Spark slightly lower than RunPod (a plausible
    aarch64 gap) so the comparison logic is exercised in CI without a GPU."""
    spark = {"task": "code", "model": "Qwen/Qwen2.5-7B-Instruct",
             "base": {"passAt1": 0.0}, "adapterScore": {"passAt1": 0.083}}
    runpod = {"task": "code", "model": "Qwen/Qwen2.5-7B-Instruct",
              "base": {"passAt1": 0.0}, "adapterScore": {"passAt1": 0.10}}
    return spark, runpod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-report", default=None, help="Spark (--local) eval_rlvr_adapter report")
    ap.add_argument("--runpod-report", default=None, help="RunPod (x86) eval_rlvr_adapter report")
    ap.add_argument("--mock", action="store_true", help="synthesize two reports (CI logic check, no GPU)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.mock:
        local, runpod = _mock_reports()
    else:
        if not args.local_report or not args.runpod_report:
            raise SystemExit("need --local-report + --runpod-report (or --mock)")
        local = json.loads(Path(args.local_report).read_text(encoding="utf-8"))
        runpod = json.loads(Path(args.runpod_report).read_text(encoding="utf-8"))

    result = compare(local, runpod)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    d = result["divergence"]
    print(f"wrote {out}")
    print(f"Spark Δ={result['spark']['delta']}  RunPod Δ={result['runpod']['delta']}  "
          f"adapter-divergence={d['adapterAbsolute']}  delta-divergence={d['deltaAbsolute']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
