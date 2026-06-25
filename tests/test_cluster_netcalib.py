#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Network-tax calibration + NCCL-benchmark command-builder tests (pure stdlib)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cluster.netcalib import (
    DEFAULT_SIZE_BYTES,
    LinkTier,
    bus_bandwidth_gbps,
    default_modeled,
    fit_tax,
    from_nccl_report,
    load_calibration,
    ring_allreduce_time_s,
)
from tools.bench_nccl_allreduce import build_torchrun_cmd, parse_args as bench_args


def test_ring_allreduce_monotonic_in_bandwidth() -> None:
    fast = ring_allreduce_time_s(DEFAULT_SIZE_BYTES, 8, 400.0, 2.0)
    slow = ring_allreduce_time_s(DEFAULT_SIZE_BYTES, 8, 25.0, 8.0)
    assert slow > fast > 0.0
    assert ring_allreduce_time_s(DEFAULT_SIZE_BYTES, 1, 400.0) == 0.0  # nothing to reduce


def test_busbw_inverts_time() -> None:
    # busbw computed from a time should round-trip the bandwidth that produced that time.
    bw = 300.0
    t = ring_allreduce_time_s(DEFAULT_SIZE_BYTES, 8, bw, latency_us=0.0)
    got = bus_bandwidth_gbps(DEFAULT_SIZE_BYTES, 8, t)
    assert abs(got - bw) < 1.0


def test_fit_tax_ordering() -> None:
    # Slower cross-node link → larger node_tax than island_tax; both non-negative.
    calib = default_modeled()
    assert calib.node_tax > calib.island_tax >= 0.0
    assert calib.source == "modeled"
    assert calib.provenance["nic"] == "modeled"


def test_comm_fraction_scales_tax() -> None:
    lo = default_modeled(comm_fraction=0.1)
    hi = default_modeled(comm_fraction=0.4)
    assert hi.node_tax > lo.node_tax
    # tax is linear in comm_fraction
    assert abs(hi.node_tax / lo.node_tax - 4.0) < 1e-6


def test_from_nccl_report_marks_measured() -> None:
    report = {
        "n_ranks": 8,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "results": [
            {"size_bytes": 256 * 1024 * 1024, "time_s": 0.0012, "algbw_gbps": 0.0, "busbw_gbps": 350.0},
        ],
    }
    calib = from_nccl_report(report)
    assert calib.source == "measured"
    assert calib.tiers["nvlink"].source == "measured"
    assert abs(calib.tiers["nvlink"].bandwidth_gbps - 350.0) < 1e-6
    assert calib.tiers["nic"].source == "modeled"  # cross-node can't be measured on one pod


def test_save_load_roundtrip(tmp_path_factory=None) -> None:
    calib = default_modeled()
    out = ROOT / "cluster" / "netcalib.json"
    assert out.exists(), "committed modeled calibration should be present"
    loaded = load_calibration(out)
    assert loaded is not None
    assert abs(loaded.node_tax - calib.node_tax) < 1e-6
    assert abs(loaded.island_tax - calib.island_tax) < 1e-6


def test_calibration_drives_simulator() -> None:
    # The simulator must pick up calibrated coefficients when tax args are None.
    from cluster.simulator import calibrated_taxes
    loaded = load_calibration()
    assert loaded is not None
    ci, cn = calibrated_taxes()
    assert abs(ci - loaded.island_tax) < 1e-9
    assert abs(cn - loaded.node_tax) < 1e-9


def test_torchrun_command_builder() -> None:
    cmd = build_torchrun_cmd(4, "out.json", sizes_mb=[1, 16, 256], iters=10)
    assert "--nproc_per_node=4" in cmd
    assert "tools/bench_nccl_allreduce.py" in cmd
    assert "--sizes-mb 1,16,256" in cmd
    assert "--iters 10" in cmd


def test_bench_dryrun_args_parse() -> None:
    args = bench_args(["--dry-run", "--gpus", "8"])
    assert args.gpus == 8 and args.dry_run and not args.run


def test_ssh_endpoint_parse_and_remote_script() -> None:
    from tools.runpod_nccl_bench import _parse_ssh_endpoint, _remote_bench_script, parse_args as rp_args
    assert _parse_ssh_endpoint("root@103.207.149.84:12226") == ("root", "103.207.149.84", 12226)
    assert _parse_ssh_endpoint("1.2.3.4") == ("root", "1.2.3.4", 22)
    # existing-pod mode clones the repo on the pod and runs torchrun across gpu-count ranks
    args = rp_args(["--ssh-endpoint", "root@h:22", "--gpu-count", "2", "--branch", "b", "--dry-run"])
    args.source = "git"
    script = _remote_bench_script(args)
    assert "git clone" in script and "--branch b" in script
    assert "--nproc_per_node=2" in script and "bench_nccl_allreduce.py --run" in script


def main() -> int:
    test_ring_allreduce_monotonic_in_bandwidth()
    test_busbw_inverts_time()
    test_fit_tax_ordering()
    test_comm_fraction_scales_tax()
    test_from_nccl_report_marks_measured()
    test_save_load_roundtrip()
    test_calibration_drives_simulator()
    test_torchrun_command_builder()
    test_bench_dryrun_args_parse()
    test_ssh_endpoint_parse_and_remote_script()
    print("test_cluster_netcalib: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
