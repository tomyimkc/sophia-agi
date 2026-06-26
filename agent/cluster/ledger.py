# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Incident & MTTR ledger (R1: 生命周期管理 / measured MTTR).

An append-only JSONL ledger of incidents, mirroring ``agent/memory.py``'s pattern.
Each incident records its lifecycle timestamps (detected → diagnosed → recovered) and
the signal that triggered it, so **MTTR is measured, not asserted** — the same
evidence discipline the repo uses for benchmark claims.

The ledger is event-sourced: ``record_*`` appends events; ``load_incidents`` folds
them into current incident state. This makes it safe for concurrent appenders and
trivially auditable (the raw event log is the source of truth).

Timestamps are injected (ISO-8601 strings) so callers and tests stay deterministic —
no hidden clock. ``mttr_stats`` derives count, mean/median MTTR and the **self-heal
ratio** (auto-resolved vs. escalated), which R4 optimises against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

# Default ledger location (gitignored at runtime; tests pass their own path).
DEFAULT_LEDGER = Path("data/cluster/incidents.jsonl")


def utcnow_iso() -> str:
    """Current UTC time as ISO-8601 (the one place a real clock is read)."""

    return datetime.now(timezone.utc).isoformat()


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


@dataclass
class Incident:
    """Folded current state of one incident."""

    incident_id: str
    node_id: str
    kind: str                       # signal / fault class, e.g. "xid_errors"
    severity: str                   # WARN | FAIL
    detected_at: str
    root_cause: str | None = None
    diagnosed_at: str | None = None
    remediation: str | None = None
    action: str | None = None
    recovered_at: str | None = None
    auto_healed: bool | None = None
    escalated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.recovered_at is None

    def mttr_seconds(self) -> float | None:
        d, r = _parse(self.detected_at), _parse(self.recovered_at)
        if d and r:
            return max(0.0, (r - d).total_seconds())
        return None

    def ttd_seconds(self) -> float | None:
        """Time-to-diagnose (detected → diagnosed)."""

        d, g = _parse(self.detected_at), _parse(self.diagnosed_at)
        if d and g:
            return max(0.0, (g - d).total_seconds())
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "severity": self.severity,
            "detected_at": self.detected_at,
            "root_cause": self.root_cause,
            "diagnosed_at": self.diagnosed_at,
            "remediation": self.remediation,
            "action": self.action,
            "recovered_at": self.recovered_at,
            "auto_healed": self.auto_healed,
            "escalated": self.escalated,
            "open": self.open,
            "mttr_seconds": self.mttr_seconds(),
        }


def _append(ledger: Path, event: dict[str, Any]) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _incident_id(node_id: str, kind: str, detected_at: str) -> str:
    """Stable id so a recurring fault on a node is one incident per detection."""

    return f"{node_id}:{kind}:{detected_at}"


def record_detection(
    *,
    node_id: str,
    kind: str,
    severity: str,
    detected_at: str | None = None,
    ledger: Path = DEFAULT_LEDGER,
) -> str:
    """Open an incident. Returns its incident_id."""

    detected_at = detected_at or utcnow_iso()
    iid = _incident_id(node_id, kind, detected_at)
    _append(ledger, {
        "event": "detected", "incident_id": iid, "node_id": node_id,
        "kind": kind, "severity": severity, "ts": detected_at,
    })
    return iid


def record_diagnosis(
    incident_id: str, *, root_cause: str, action: str, remediation: str,
    diagnosed_at: str | None = None, ledger: Path = DEFAULT_LEDGER,
) -> None:
    _append(ledger, {
        "event": "diagnosed", "incident_id": incident_id, "root_cause": root_cause,
        "action": action, "remediation": remediation, "ts": diagnosed_at or utcnow_iso(),
    })


def record_recovery(
    incident_id: str, *, auto_healed: bool, recovered_at: str | None = None,
    note: str | None = None, ledger: Path = DEFAULT_LEDGER,
) -> None:
    _append(ledger, {
        "event": "recovered", "incident_id": incident_id, "auto_healed": auto_healed,
        "note": note, "ts": recovered_at or utcnow_iso(),
    })


def record_escalation(
    incident_id: str, *, reason: str, ts: str | None = None, ledger: Path = DEFAULT_LEDGER,
) -> None:
    _append(ledger, {
        "event": "escalated", "incident_id": incident_id, "reason": reason,
        "ts": ts or utcnow_iso(),
    })


def load_incidents(ledger: Path = DEFAULT_LEDGER) -> list[Incident]:
    """Fold the event log into current incident state."""

    if not Path(ledger).exists():
        return []
    incidents: dict[str, Incident] = {}
    for line in Path(ledger).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        iid = ev.get("incident_id")
        if not iid:
            continue
        kind = ev.get("event")
        if kind == "detected":
            incidents[iid] = Incident(
                incident_id=iid, node_id=ev["node_id"], kind=ev["kind"],
                severity=ev.get("severity", "FAIL"), detected_at=ev["ts"],
            )
        elif iid in incidents:
            inc = incidents[iid]
            if kind == "diagnosed":
                inc.root_cause = ev.get("root_cause")
                inc.action = ev.get("action")
                inc.remediation = ev.get("remediation")
                inc.diagnosed_at = ev.get("ts")
            elif kind == "recovered":
                inc.recovered_at = ev.get("ts")
                inc.auto_healed = bool(ev.get("auto_healed"))
                if ev.get("note"):
                    inc.notes.append(ev["note"])
            elif kind == "escalated":
                inc.escalated = True
                if ev.get("reason"):
                    inc.notes.append(f"escalated: {ev['reason']}")
    return list(incidents.values())


def mttr_stats(ledger: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    """Aggregate MTTR + self-heal ratio over the ledger.

    Returns counts, mean/median MTTR (seconds, over *recovered* incidents) and the
    self-heal ratio (auto-resolved / recovered). All-zero when the ledger is empty.
    """

    incidents = load_incidents(ledger)
    recovered = [i for i in incidents if not i.open]
    mttrs = [m for i in recovered if (m := i.mttr_seconds()) is not None]
    ttds = [t for i in recovered if (t := i.ttd_seconds()) is not None]
    auto = sum(1 for i in recovered if i.auto_healed)
    escalated = sum(1 for i in incidents if i.escalated)
    return {
        "total": len(incidents),
        "open": sum(1 for i in incidents if i.open),
        "recovered": len(recovered),
        "auto_healed": auto,
        "escalated": escalated,
        "self_heal_ratio": (auto / len(recovered)) if recovered else 0.0,
        "mttr_seconds_mean": float(mean(mttrs)) if mttrs else 0.0,
        "mttr_seconds_median": float(median(mttrs)) if mttrs else 0.0,
        "ttd_seconds_mean": float(mean(ttds)) if ttds else 0.0,
    }
