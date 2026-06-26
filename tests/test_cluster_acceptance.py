# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the fail-closed bring-up acceptance gate (R2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster.acceptance import (  # noqa: E402
    accept_node,
    baseline_for,
    load_baselines,
    mock_benchmark_runner,
)

H100 = "NVIDIA H100 80GB HBM3"


def test_healthy_node_accepted():
    res = accept_node("gpu-node-000", H100, mock_benchmark_runner)
    assert res.accepted
    assert res.failures() == []


def test_regressed_node_rejected():
    # odd-ending id injects an NCCL + HBM regression below the H100 floor
    res = accept_node("gpu-node-001", H100, mock_benchmark_runner)
    assert not res.accepted
    failed = {c.metric for c in res.failures()}
    assert "min_nccl_allreduce_busbw_gbps" in failed
    assert "min_hbm_bandwidth_gbps" in failed


def test_unmeasured_metric_fails_closed():
    def partial_runner(node_id, gpu_model):
        out = mock_benchmark_runner(node_id, gpu_model)
        out.pop("nccl_allreduce_busbw_gbps")  # drop a measurement entirely
        return out
    res = accept_node("gpu-node-000", H100, partial_runner)
    assert not res.accepted
    miss = next(c for c in res.failures() if c.metric == "min_nccl_allreduce_busbw_gbps")
    assert miss.measured is None
    assert "fail-closed" in miss.detail


def test_baseline_merges_type_over_defaults():
    baselines = load_baselines()
    floor = baseline_for(H100, baselines)
    # H100 overrides the default HBM floor upward.
    assert floor["min_hbm_bandwidth_gbps"] == 2800.0
    # A default-only key survives.
    assert "max_p2p_latency_us" in floor


def test_unknown_gpu_uses_defaults():
    baselines = load_baselines()
    floor = baseline_for("Some Unknown GPU", baselines)
    assert floor["min_hbm_bandwidth_gbps"] == baselines["defaults"]["min_hbm_bandwidth_gbps"]
