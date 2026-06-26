# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Prometheus exporter rendering and calibrated alert thresholds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster.calibrate import fit_threshold  # noqa: E402
from agent.cluster.provider import MockProvider, sweep  # noqa: E402
from services.cluster_exporter.main import render_metrics  # noqa: E402


def test_render_metrics_is_valid_prometheus_text():
    nodes = sweep(MockProvider(size=6))
    mttr = {"total": 3, "open": 1, "mttr_seconds_mean": 1800.0, "self_heal_ratio": 0.5}
    text = render_metrics(nodes, mttr, acceptance={"gpu-node-000": True, "gpu-node-001": False})

    # HELP/TYPE headers present for a representative metric
    assert "# HELP sophia_node_health" in text
    assert "# TYPE sophia_node_health gauge" in text
    # rollups reflect the synthetic fleet (which contains a FAIL node)
    assert "sophia_fleet_nodes_total 6" in text
    assert "sophia_selfheal_ratio 0.5" in text
    assert 'sophia_node_acceptance_pass{node="gpu-node-001"} 0' in text
    # absent telemetry is omitted, not zero-filled (the unreachable node has no temp line)
    assert 'sophia_gpu_temp_celsius{node="gpu-node-005"}' not in text


def test_render_omits_unmeasured_telemetry():
    from agent.cluster.health import NodeMetrics
    nodes = [NodeMetrics(node_id="n", reachable=True)]  # everything unknown
    text = render_metrics(nodes, {"total": 0})
    assert "sophia_gpu_temp_celsius{" not in text  # nothing measured → no sample
    assert "sophia_node_health{" in text            # verdict always emitted


def test_calibration_fit_separable_signal():
    # hotter → fault, cleanly separable above 85
    values = [60.0 + i for i in range(40)]
    labels = [v >= 85 for v in values]
    fit = fit_threshold("gpu_temp_c", values, labels, max_false_alert_rate=0.05, min_faults=8)
    assert fit.threshold is not None
    assert fit.recall >= 0.9
    assert fit.false_alert_rate <= 0.05
    assert fit.adopted  # enough faults present


def test_calibration_small_n_not_adopted():
    values = [60.0, 90.0]
    labels = [False, True]
    fit = fit_threshold("gpu_temp_c", values, labels, min_faults=8)
    assert not fit.adopted
    assert "NOT adopted" in fit.note


def test_calibration_no_faults_returns_unfit():
    fit = fit_threshold("gpu_temp_c", [60.0, 61.0], [False, False])
    assert fit.threshold is None and not fit.adopted
