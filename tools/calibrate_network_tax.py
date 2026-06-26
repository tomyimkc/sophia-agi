#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibrate the simulator's network-tax coefficients from all-reduce cost.

Two modes:
  (default) MODELED — derive island_tax/node_tax from documented interconnect line rates
                      (NVLink / NVSwitch / RoCE), writing a reproducible cluster/netcalib.json.
  --from-nccl FILE   MEASURED — ingest a tools/bench_nccl_allreduce.py report and replace the
                      NVLink tier with the measured bus bandwidth (cross-node stays modeled).

The written cluster/netcalib.json is consumed automatically by cluster/simulator.py, so a
fresh calibration flows straight into the scheduler and fault simulations.

    python tools/calibrate_network_tax.py --markdown                      # modeled
    python tools/calibrate_network_tax.py --from-nccl agi-proof/benchmark-results/cluster/nccl-allreduce.public-report.json
    python tools/calibrate_network_tax.py --comm-fraction 0.4 --n-ranks 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.netcalib import (
    DEFAULT_CALIB_PATH,
    DEFAULT_COMM_FRACTION,
    DEFAULT_N_RANKS,
    default_modeled,
    from_nccl_report,
    ring_allreduce_time_s,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-nccl", default=None, help="bench_nccl_allreduce.py report JSON")
    p.add_argument("--comm-fraction", type=float, default=DEFAULT_COMM_FRACTION,
                   help="comm share of a training step at full NVLink speed")
    p.add_argument("--n-ranks", type=int, default=DEFAULT_N_RANKS)
    p.add_argument("--out", default=str(DEFAULT_CALIB_PATH))
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--no-write", action="store_true", help="compute + print only")
    return p.parse_args(argv)


def run(args: argparse.Namespace):
    if args.from_nccl:
        report = json.loads(Path(args.from_nccl).read_text(encoding="utf-8"))
        calib = from_nccl_report(report, comm_fraction=args.comm_fraction)
    else:
        calib = default_modeled(comm_fraction=args.comm_fraction, n_ranks=args.n_ranks)
    return calib


def _markdown(calib) -> str:
    lines = [
        f"**source:** {calib.source}  ·  **comm_fraction:** {calib.comm_fraction}  "
        f"·  **size:** {calib.size_bytes // (1024*1024)} MB  ·  **ranks:** {calib.n_ranks}",
        "",
        "| tier | bandwidth (GB/s) | latency (us) | all-reduce (ms) | source |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, tier in calib.tiers.items():
        t_ms = ring_allreduce_time_s(calib.size_bytes, calib.n_ranks,
                                     tier.bandwidth_gbps, tier.latency_us) * 1e3
        lines.append(f"| {name} | {tier.bandwidth_gbps:.1f} | {tier.latency_us:.1f} "
                     f"| {t_ms:.2f} | {tier.source} |")
    lines += [
        "",
        f"**island_tax = {calib.island_tax:.4f}**  ·  **node_tax = {calib.node_tax:.4f}**",
        f"_{calib.notes}_",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    calib = run(args)
    if not args.no_write:
        path = calib.save(args.out)
        print(f"wrote {path}")
    print(_markdown(calib) if args.markdown else json.dumps(calib.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
