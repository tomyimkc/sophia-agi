# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia AI compute-cluster reliability & performance layer.

A small, offline-testable toolkit that turns the repo's existing GPU/RunPod usage
into *reliability engineering*: fleet inspection (巡检), fault localization, an
MTTR-measuring incident ledger, a fail-closed bring-up acceptance gate, gated
auto-remediation (自愈), and calibrated alert thresholds.

Design follows the repo's discipline (see VISION.md):

* **Fail-closed.** Missing or ambiguous telemetry never reads as "healthy"; a node
  with unknown critical metrics is degraded, not passed.
* **Provenance for ops.** Every health verdict and every remediation cites the exact
  signal that triggered it — no unexplained reboots.
* **Calibrated, risk-proportional action.** Low-risk + high-confidence remediations
  auto-heal; high-stakes or low-confidence ones escalate to a human.
* **Offline by default.** Providers are injectable; the deterministic ``MockProvider``
  needs no network and no cost, exactly like the repo's ``--dry-run`` tools.
"""

from __future__ import annotations

from agent.cluster.health import (
    HealthReason,
    NodeHealth,
    NodeMetrics,
    Thresholds,
    Verdict,
    evaluate_node,
)

__all__ = [
    "HealthReason",
    "NodeHealth",
    "NodeMetrics",
    "Thresholds",
    "Verdict",
    "evaluate_node",
]
