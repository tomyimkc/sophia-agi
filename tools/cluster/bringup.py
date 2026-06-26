#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Node bring-up acceptance gate (R2: 交付前基线检查与性能验证).

Run the acceptance suite against a node and gate it fail-closed against the committed
per-GPU baseline (``config/cluster_baselines.json``). A node is accepted only if every
measured metric clears its floor; an unmeasured metric fails the gate.

    # offline demo with the deterministic mock benchmark runner
    python3 tools/cluster/bringup.py --node gpu-node-000 --gpu "NVIDIA H100 80GB HBM3"
    python3 tools/cluster/bringup.py --node gpu-node-001 --json   # injects a regression

Exit code is 0 only when the node is accepted, so this slots into a delivery pipeline.
A live runner (dcgmi/GEMM/NCCL/ib_write_bw over the RunPod SSH lifecycle) can be wired
in place of the mock without touching the gating logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster.acceptance import accept_node, load_baselines, mock_benchmark_runner  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sophia node bring-up acceptance gate (R2).")
    ap.add_argument("--node", required=True, help="node id to accept")
    ap.add_argument("--gpu", default="NVIDIA H100 80GB HBM3", help="GPU model (selects baseline)")
    ap.add_argument("--baselines", default=None, help="path to baselines JSON (default: config/)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    baselines = load_baselines(args.baselines) if args.baselines else load_baselines()
    # Offline mock runner; swap for a live SSH-driven runner in production.
    result = accept_node(args.node, args.gpu, mock_benchmark_runner, baselines=baselines)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        status = "ACCEPTED" if result.accepted else "REJECTED"
        print(f"[{status}] {result.node_id}  ({result.gpu_model})")
        for c in result.checks:
            mark = "ok " if c.passed else "XX "
            print(f"  {mark} {c.label:<34} {c.detail}")
        if not result.accepted:
            print(f"\n  rejected on: {', '.join(f.label for f in result.failures())}")

    return 0 if result.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
