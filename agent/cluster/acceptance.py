# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Bring-up acceptance gate (R2: 交付前的基线检查与性能验证).

A node is declared production-ready only when **every** measured benchmark clears its
committed baseline for that GPU type. This is the hardware analogue of the repo's
``agent/gate.py``: fail-closed, opt-out-never. A missing measurement does not pass —
it fails the gate ("unverified ⇒ not accepted"), exactly like an ungrounded claim.

The benchmark *runner* is injectable:

* ``mock_benchmark_runner`` returns deterministic synthetic numbers for offline tests
  and demos (no GPU, no cost).
* A live runner would execute the suite over the existing RunPod SSH lifecycle
  (``tools/runpod_rlvr.py``): ``dcgmi diag``, a cuBLAS GEMM, an HBM bandwidth probe,
  ``p2pBandwidthLatencyTest``, an NCCL all-reduce, and ``ib_write_bw`` for RDMA.

The gating logic here is the contribution; the measurement plumbing is swappable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
BASELINES_PATH = ROOT / "config" / "cluster_baselines.json"

# A benchmark runner takes (node_id, gpu_model) and returns measured metric values.
BenchmarkRunner = Callable[[str, str], dict[str, float]]

# Each check: metric key, human label, comparison ('min' = measured must be ≥ floor,
# 'max' = measured must be ≤ ceiling). Keys match config/cluster_baselines.json.
CHECKS: tuple[tuple[str, str, str], ...] = (
    ("min_gpu_count", "GPU count", "min"),
    ("max_gpu_temp_idle_c", "Idle GPU temp (°C)", "max"),
    ("max_ecc_uncorrectable", "Uncorrectable ECC", "max"),
    ("min_hbm_bandwidth_gbps", "HBM bandwidth (GB/s)", "min"),
    ("min_gemm_tflops_bf16", "GEMM BF16 (TFLOPS)", "min"),
    ("min_nvlink_bandwidth_gbps", "NVLink bandwidth (GB/s)", "min"),
    ("min_nccl_allreduce_busbw_gbps", "NCCL all-reduce busbw (GB/s)", "min"),
    ("min_rdma_ib_write_bw_gbps", "RDMA ib_write_bw (GB/s)", "min"),
    ("max_p2p_latency_us", "P2P latency (µs)", "max"),
)

# Map a baseline key to the measured-metric key the runner reports.
MEASURED_KEY = {
    "min_gpu_count": "gpu_count",
    "max_gpu_temp_idle_c": "gpu_temp_idle_c",
    "max_ecc_uncorrectable": "ecc_uncorrectable",
    "min_hbm_bandwidth_gbps": "hbm_bandwidth_gbps",
    "min_gemm_tflops_bf16": "gemm_tflops_bf16",
    "min_nvlink_bandwidth_gbps": "nvlink_bandwidth_gbps",
    "min_nccl_allreduce_busbw_gbps": "nccl_allreduce_busbw_gbps",
    "min_rdma_ib_write_bw_gbps": "rdma_ib_write_bw_gbps",
    "max_p2p_latency_us": "p2p_latency_us",
}


def load_baselines(path: Path | str = BASELINES_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline_for(gpu_model: str | None, baselines: dict[str, Any]) -> dict[str, float]:
    """Merge type-specific baseline over defaults (type wins where it overrides)."""

    merged = dict(baselines.get("defaults", {}))
    types = baselines.get("gpu_types", {})
    if gpu_model and gpu_model in types:
        merged.update(types[gpu_model])
    return {k: v for k, v in merged.items() if not str(k).startswith("_")}


@dataclass(frozen=True)
class CheckResult:
    metric: str
    label: str
    measured: float | None
    threshold: float
    comparison: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric, "label": self.label, "measured": self.measured,
            "threshold": self.threshold, "comparison": self.comparison,
            "passed": self.passed, "detail": self.detail,
        }


@dataclass
class AcceptanceResult:
    node_id: str
    gpu_model: str | None
    accepted: bool
    checks: list[CheckResult] = field(default_factory=list)

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "gpu_model": self.gpu_model,
            "accepted": self.accepted,
            "checks": [c.to_dict() for c in self.checks],
            "failures": [c.label for c in self.failures()],
        }


def _evaluate_check(metric: str, label: str, comparison: str,
                    threshold: float, measured: float | None) -> CheckResult:
    # Fail-closed: an unmeasured metric never passes the gate.
    if measured is None:
        return CheckResult(metric, label, None, threshold, comparison, False,
                           "not measured — fail-closed (unverified ⇒ not accepted)")
    if comparison == "min":
        ok = measured >= threshold
        detail = f"{measured} {'≥' if ok else '<'} floor {threshold}"
    else:  # 'max'
        ok = measured <= threshold
        detail = f"{measured} {'≤' if ok else '>'} ceiling {threshold}"
    return CheckResult(metric, label, measured, threshold, comparison, ok, detail)


def accept_node(
    node_id: str,
    gpu_model: str | None,
    runner: BenchmarkRunner,
    *,
    baselines: dict[str, Any] | None = None,
) -> AcceptanceResult:
    """Run the acceptance suite and gate the node fail-closed."""

    baselines = baselines or load_baselines()
    floor = baseline_for(gpu_model, baselines)
    measured = runner(node_id, gpu_model or "")
    checks: list[CheckResult] = []
    for metric, label, comparison in CHECKS:
        if metric not in floor:
            continue
        mkey = MEASURED_KEY[metric]
        checks.append(_evaluate_check(metric, label, comparison, float(floor[metric]),
                                      measured.get(mkey)))
    accepted = all(c.passed for c in checks) and len(checks) > 0
    return AcceptanceResult(node_id, gpu_model, accepted, checks)


def mock_benchmark_runner(node_id: str, gpu_model: str) -> dict[str, float]:
    """Deterministic synthetic benchmark output for offline tests/demos.

    Node ids ending in an even digit pass; odd-ending ids inject a regression
    (NCCL busbw + HBM below floor) so the gate has both outcomes to exercise.
    """

    healthy = {
        "gpu_count": 8.0,
        "gpu_temp_idle_c": 38.0,
        "ecc_uncorrectable": 0.0,
        "hbm_bandwidth_gbps": 3050.0,
        "gemm_tflops_bf16": 760.0,
        "nvlink_bandwidth_gbps": 740.0,
        "nccl_allreduce_busbw_gbps": 245.0,
        "rdma_ib_write_bw_gbps": 97.0,
        "p2p_latency_us": 8.5,
    }
    last = node_id.strip()[-1:] or "0"
    if last.isdigit() and int(last) % 2 == 1:
        regressed = dict(healthy)
        regressed["nccl_allreduce_busbw_gbps"] = 140.0  # below H100 floor 230
        regressed["hbm_bandwidth_gbps"] = 2400.0        # below H100 floor 2800
        return regressed
    return healthy
