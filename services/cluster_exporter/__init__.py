# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia cluster Prometheus exporter (R3: 监控 / 可观测性).

Renders the fleet inspection sweep (agent/cluster) and incident-ledger MTTR stats as
Prometheus text-format metrics. Pure stdlib (no prometheus_client dependency) so it
runs airgapped, matching the repo's local-first stance.
"""

from __future__ import annotations

from services.cluster_exporter.main import render_metrics

__all__ = ["render_metrics"]
