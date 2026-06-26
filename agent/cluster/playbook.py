# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fault-localization playbook (R1: 故障定位).

Maps a node's failing signal to a root-cause hypothesis, a proposed remediation, a
risk level and a confidence. Every diagnosis cites the triggering signal — so an
operator (or the gated auto-healer in ``heal.py``) can see *why* an action is being
proposed. Confidence is deliberately conservative: signals with a single unambiguous
cause (GPU fell off bus) score high; broad signals (high temp) score lower because
they have several possible causes.

This is a lookup-style decision tree, not an LLM — it is fully deterministic and
auditable. Sophia's reasoning core can sit *on top* of it for ambiguous multi-signal
cases, but the base layer never needs a model to fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.cluster.health import NodeHealth, Verdict


class Risk:
    """Coarse risk tiers for remediations (string constants, stable for ledgers)."""

    LOW = "low"        # non-disruptive (clear counters, GPU reset on an idle node)
    MEDIUM = "medium"  # disruptive to one node (drain + reboot)
    HIGH = "high"      # fleet-affecting or data-risking (RMA, fabric reconfig)


@dataclass(frozen=True)
class Diagnosis:
    """A single root-cause hypothesis with its proposed fix."""

    signal: str
    root_cause: str
    remediation: str
    action: str          # machine-readable action key for heal.py
    risk: str
    confidence: float    # 0..1, how sure we are of the root cause
    evidence: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "root_cause": self.root_cause,
            "remediation": self.remediation,
            "action": self.action,
            "risk": self.risk,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }


# signal -> diagnosis builder. Keyed on the same signal names the evaluator emits.
def _diagnose_signal(signal: str, value: Any) -> Diagnosis:
    if signal == "reachability":
        return Diagnosis(signal, "Node offline: host down, SSH/agent dead, or network partition",
                         "Ping/console-check; if host is up, restart the node agent; else power-cycle and re-bring-up",
                         action="cordon_and_investigate", risk=Risk.MEDIUM, confidence=0.70, evidence=value)
    if signal == "xid_errors":
        vals = value if isinstance(value, list) else [value]
        if 79 in vals:
            return Diagnosis(signal, "XID 79: GPU has fallen off the bus (PCIe link lost)",
                             "Drain the node and power-cycle; if it recurs, reseat/RMA the GPU",
                             action="drain_and_reboot", risk=Risk.MEDIUM, confidence=0.90, evidence=vals)
        if any(x in (48, 94, 95) for x in vals):
            return Diagnosis(signal, "Fatal ECC XID: double-bit / uncontained ECC fault",
                             "Drain the node; run dcgmi diag; row-remap or RMA the GPU before returning to prod",
                             action="drain_and_diag", risk=Risk.HIGH, confidence=0.85, evidence=vals)
        if 74 in vals:
            return Diagnosis(signal, "XID 74: NVLink error on the GPU fabric",
                             "Drain; check NVLink topology (nvidia-smi nvlink -s); reseat if a single link is bad",
                             action="drain_and_diag", risk=Risk.MEDIUM, confidence=0.80, evidence=vals)
        return Diagnosis(signal, f"Non-fatal XID(s) {vals}: transient or app-induced",
                         "Capture nvidia-smi -q and dmesg; watch for recurrence before acting",
                         action="observe", risk=Risk.LOW, confidence=0.55, evidence=vals)
    if signal == "ecc_uncorrectable":
        return Diagnosis(signal, "Uncorrectable ECC errors on GPU memory",
                         "Drain; run dcgmi diag -r 3; trigger row-remap; RMA if remap is exhausted",
                         action="drain_and_diag", risk=Risk.HIGH, confidence=0.85, evidence=value)
    if signal == "rdma_link_down":
        return Diagnosis(signal, "RDMA/IB link(s) down: node isolated from the training fabric",
                         "Check ibstat / link LEDs; restart the IB interface; reseat cable if link won't come up",
                         action="restart_fabric_iface", risk=Risk.MEDIUM, confidence=0.80, evidence=value)
    if signal == "nvlink_down":
        return Diagnosis(signal, "NVLink lane(s) down: degraded intra-node GPU bandwidth",
                         "Drain; inspect nvlink status; reseat GPU; RMA if a bridge is faulty",
                         action="drain_and_diag", risk=Risk.MEDIUM, confidence=0.75, evidence=value)
    if signal == "gpu_temp_c":
        return Diagnosis(signal, "GPU over-temperature: cooling fault, dust, or hot aisle",
                         "Check inlet temp & fan/pump; clean airflow; cap power (nvidia-smi -pl) until cooling is fixed",
                         action="powercap_and_inspect_cooling", risk=Risk.LOW, confidence=0.55, evidence=value)
    if signal == "throttled":
        return Diagnosis(signal, "Clock throttling (thermal/power): perf loss, not yet a hard fault",
                         "Correlate with temp/power; if thermal, address cooling; if power, check PSU/power budget",
                         action="observe", risk=Risk.LOW, confidence=0.50, evidence=value)
    if signal == "disk_used_frac":
        return Diagnosis(signal, "Disk near full: checkpoints/logs will fail writes",
                         "Garbage-collect old checkpoints/logs; expand volume; alert the job owner",
                         action="gc_disk", risk=Risk.LOW, confidence=0.80, evidence=value)
    if signal == "mem_used_frac":
        return Diagnosis(signal, "Host memory pressure: OOM-kill risk for the training process",
                         "Identify the offending process; check for a leak; reduce dataloader workers / pinned mem",
                         action="observe", risk=Risk.LOW, confidence=0.55, evidence=value)
    # Unknown-but-degraded (fail-closed WARN from a missing metric).
    return Diagnosis(signal, f"Telemetry gap on '{signal}': cannot confirm health",
                     "Restore the node's DCGM/nvidia-smi agent so the metric reports again",
                     action="restore_telemetry", risk=Risk.LOW, confidence=0.40, evidence=value)


def diagnose(health: NodeHealth) -> list[Diagnosis]:
    """Return one diagnosis per non-PASS reason, worst-first by reason verdict.

    A PASS node returns an empty list. Diagnoses preserve the order of the failing
    reasons so the most severe signals are addressed first.
    """

    bad = [r for r in health.reasons if r.verdict != Verdict.PASS]
    bad.sort(key=lambda r: r.verdict, reverse=True)  # FAIL before WARN
    return [_diagnose_signal(r.signal, r.value) for r in bad]


def primary_diagnosis(health: NodeHealth) -> Diagnosis | None:
    """The single most severe, most confident diagnosis (what heal.py acts on)."""

    diags = diagnose(health)
    if not diags:
        return None
    risk_rank = {Risk.LOW: 0, Risk.MEDIUM: 1, Risk.HIGH: 2}
    # Worst risk first; among equal risk, the most confident hypothesis.
    diags.sort(key=lambda d: (risk_rank.get(d.risk, 0), d.confidence), reverse=True)
    return diags[0]
