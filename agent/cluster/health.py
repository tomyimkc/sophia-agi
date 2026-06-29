# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Node health model + fail-closed verdict logic (R1: 巡检 / fault detection).

``evaluate_node`` turns a ``NodeMetrics`` telemetry snapshot into a ``NodeHealth``
verdict (PASS / WARN / FAIL) plus a list of ``HealthReason`` records. The verdict is
the *worst* reason — and crucially, **fail-closed**: when a node is reachable but a
critical metric is missing (``None``), that is treated as a WARN ("unknown, can't
clear it"), never silently passed. Each reason cites the signal name and the observed
value, so a verdict is auditable — provenance for ops.

The XID classification follows NVIDIA's published fatal codes (GPU fell off bus,
double-bit ECC, NVLink errors, row-remap failures); anything not known-fatal is a
WARN rather than being ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Verdict(IntEnum):
    """Ordered so ``max(...)`` yields the worst verdict across reasons."""

    PASS = 0
    WARN = 1
    FAIL = 2

    @property
    def label(self) -> str:
        return self.name


# NVIDIA XID codes that indicate hardware-fatal conditions (non-exhaustive but the
# ones an ops sweep must never miss). Sources: NVIDIA XID error documentation.
FATAL_XIDS: frozenset[int] = frozenset(
    {
        48,  # Double-bit ECC error (DBE)
        56,  # Display engine error / fatal
        57,  # Internal micro-controller halt
        62,  # Internal micro-controller halt
        63,  # ECC page retirement recording event
        64,  # ECC page retirement / row-remap failure
        74,  # NVLink error (NVLINK_ERROR)
        79,  # GPU has fallen off the bus
        92,  # High single-bit ECC error rate
        94,  # Contained ECC error
        95,  # Uncontained ECC error
    }
)


@dataclass(frozen=True)
class Thresholds:
    """Tunable health thresholds. Defaults are deliberately conservative.

    Temperatures in Celsius; fractions in [0, 1]. ``calibrate.py`` can refit the
    temperature/utilisation thresholds against real incident outcomes.
    """

    temp_warn_c: float = 80.0
    temp_fail_c: float = 88.0
    mem_warn_frac: float = 0.90
    mem_fail_frac: float = 0.97
    disk_warn_frac: float = 0.85
    disk_fail_frac: float = 0.95
    # Any uncorrectable ECC error is a FAIL; correctable-rate handling is via XID 92.
    ecc_uncorrectable_fail: int = 1


@dataclass(frozen=True)
class NodeMetrics:
    """A single telemetry snapshot for one node.

    ``None`` means "not measured / unknown". Reachable-but-unknown critical metrics
    are penalised (fail-closed), so a provider that cannot read DCGM telemetry yields
    a degraded verdict rather than a false-green one.
    """

    node_id: str
    gpu_model: str | None = None
    reachable: bool = True
    gpu_temp_c: float | None = None
    gpu_util: float | None = None  # 0..1
    mem_used_frac: float | None = None  # 0..1
    disk_used_frac: float | None = None  # 0..1
    ecc_uncorrectable: int | None = None
    xid_errors: tuple[int, ...] = ()
    throttled: bool | None = None
    nvlink_down: int | None = None  # count of down NVLink lanes
    rdma_link_down: int | None = None  # count of down RDMA/IB links
    collected_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "gpu_model": self.gpu_model,
            "reachable": self.reachable,
            "gpu_temp_c": self.gpu_temp_c,
            "gpu_util": self.gpu_util,
            "mem_used_frac": self.mem_used_frac,
            "disk_used_frac": self.disk_used_frac,
            "ecc_uncorrectable": self.ecc_uncorrectable,
            "xid_errors": list(self.xid_errors),
            "throttled": self.throttled,
            "nvlink_down": self.nvlink_down,
            "rdma_link_down": self.rdma_link_down,
            "collected_at": self.collected_at,
        }


@dataclass(frozen=True)
class HealthReason:
    """One verdict justification: which signal, how bad, and the observed value."""

    signal: str
    verdict: Verdict
    message: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "verdict": self.verdict.label,
            "message": self.message,
            "value": self.value,
        }


@dataclass(frozen=True)
class NodeHealth:
    """Aggregate verdict for a node plus the reasons that produced it."""

    node_id: str
    verdict: Verdict
    reasons: list[HealthReason]
    metrics: NodeMetrics

    @property
    def ok(self) -> bool:
        return self.verdict == Verdict.PASS

    @property
    def failed(self) -> bool:
        return self.verdict == Verdict.FAIL

    def fail_reasons(self) -> list[HealthReason]:
        return [r for r in self.reasons if r.verdict == Verdict.FAIL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "verdict": self.verdict.label,
            "reasons": [r.to_dict() for r in self.reasons],
            "metrics": self.metrics.to_dict(),
        }


def _check_unknown(value: Any, signal: str, label: str) -> HealthReason | None:
    """Fail-closed: a reachable node with an unknown critical metric is a WARN."""

    if value is None:
        return HealthReason(
            signal=signal,
            verdict=Verdict.WARN,
            message=f"{label} not reported — cannot clear node fail-closed",
            value=None,
        )
    return None


def evaluate_node(metrics: NodeMetrics, thresholds: Thresholds | None = None) -> NodeHealth:
    """Compute a fail-closed health verdict for one node.

    Returns ``NodeHealth`` whose ``verdict`` is the worst of all reasons. A node with
    no negative signals and all critical metrics present is ``PASS``.
    """

    t = thresholds or Thresholds()
    reasons: list[HealthReason] = []

    # Reachability gates everything — an unreachable node is FAIL, full stop.
    if not metrics.reachable:
        reasons.append(
            HealthReason(
                "reachability",
                Verdict.FAIL,
                "node unreachable (SSH/agent did not respond)",
                False,
            )
        )
        return NodeHealth(metrics.node_id, Verdict.FAIL, reasons, metrics)

    # Temperature.
    if (unk := _check_unknown(metrics.gpu_temp_c, "gpu_temp_c", "GPU temperature")):
        reasons.append(unk)
    elif metrics.gpu_temp_c >= t.temp_fail_c:
        reasons.append(HealthReason("gpu_temp_c", Verdict.FAIL,
                                    f"GPU temp {metrics.gpu_temp_c}°C ≥ fail {t.temp_fail_c}°C",
                                    metrics.gpu_temp_c))
    elif metrics.gpu_temp_c >= t.temp_warn_c:
        reasons.append(HealthReason("gpu_temp_c", Verdict.WARN,
                                    f"GPU temp {metrics.gpu_temp_c}°C ≥ warn {t.temp_warn_c}°C",
                                    metrics.gpu_temp_c))

    # Uncorrectable ECC — any is a FAIL.
    if (unk := _check_unknown(metrics.ecc_uncorrectable, "ecc_uncorrectable", "ECC counters")):
        reasons.append(unk)
    elif metrics.ecc_uncorrectable >= t.ecc_uncorrectable_fail:
        reasons.append(HealthReason("ecc_uncorrectable", Verdict.FAIL,
                                    f"{metrics.ecc_uncorrectable} uncorrectable ECC error(s)",
                                    metrics.ecc_uncorrectable))

    # XID errors — classify fatal vs. other.
    fatal = sorted(x for x in metrics.xid_errors if x in FATAL_XIDS)
    other = sorted(x for x in metrics.xid_errors if x not in FATAL_XIDS)
    if fatal:
        reasons.append(HealthReason("xid_errors", Verdict.FAIL,
                                    f"fatal XID(s) present: {fatal}", fatal))
    if other:
        reasons.append(HealthReason("xid_errors", Verdict.WARN,
                                    f"non-fatal XID(s) present: {other}", other))

    # NVLink / RDMA interconnect — down links break multi-GPU/multi-node training.
    if (unk := _check_unknown(metrics.nvlink_down, "nvlink_down", "NVLink status")):
        reasons.append(unk)
    elif metrics.nvlink_down > 0:
        reasons.append(HealthReason("nvlink_down", Verdict.FAIL,
                                    f"{metrics.nvlink_down} NVLink lane(s) down", metrics.nvlink_down))

    if (unk := _check_unknown(metrics.rdma_link_down, "rdma_link_down", "RDMA link status")):
        reasons.append(unk)
    elif metrics.rdma_link_down > 0:
        reasons.append(HealthReason("rdma_link_down", Verdict.FAIL,
                                    f"{metrics.rdma_link_down} RDMA/IB link(s) down", metrics.rdma_link_down))

    # Memory pressure.
    if metrics.mem_used_frac is not None:
        if metrics.mem_used_frac >= t.mem_fail_frac:
            reasons.append(HealthReason("mem_used_frac", Verdict.FAIL,
                                        f"memory {metrics.mem_used_frac:.0%} ≥ fail {t.mem_fail_frac:.0%}",
                                        metrics.mem_used_frac))
        elif metrics.mem_used_frac >= t.mem_warn_frac:
            reasons.append(HealthReason("mem_used_frac", Verdict.WARN,
                                        f"memory {metrics.mem_used_frac:.0%} ≥ warn {t.mem_warn_frac:.0%}",
                                        metrics.mem_used_frac))

    # Disk pressure (checkpoints fill disks; a full disk kills a training job).
    if metrics.disk_used_frac is not None:
        if metrics.disk_used_frac >= t.disk_fail_frac:
            reasons.append(HealthReason("disk_used_frac", Verdict.FAIL,
                                        f"disk {metrics.disk_used_frac:.0%} ≥ fail {t.disk_fail_frac:.0%}",
                                        metrics.disk_used_frac))
        elif metrics.disk_used_frac >= t.disk_warn_frac:
            reasons.append(HealthReason("disk_used_frac", Verdict.WARN,
                                        f"disk {metrics.disk_used_frac:.0%} ≥ warn {t.disk_warn_frac:.0%}",
                                        metrics.disk_used_frac))

    # Thermal/power throttling.
    if metrics.throttled is True:
        reasons.append(HealthReason("throttled", Verdict.WARN,
                                    "GPU is clock-throttled (thermal/power)", True))

    verdict = max((r.verdict for r in reasons), default=Verdict.PASS)
    return NodeHealth(metrics.node_id, Verdict(verdict), reasons, metrics)
