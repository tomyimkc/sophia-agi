#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia cluster reliability loop in 30 seconds — end to end, offline.

Walks the whole AI-compute-cluster reliability layer the way an on-call engineer would:

    bring-up acceptance  ->  fleet 巡检 (inspection)  ->  inject + localize a fault
        ->  gated remediation (auto-heal vs human escalate)  ->  measured MTTR / self-heal

    python scripts/demo_cluster_loop.py

No network, no GPUs, no API key (deterministic MockProvider + injected runners + a
temp ledger). Mirrors the discipline of scripts/demo_gate.py: every action is
fail-closed and every number is measured, not asserted.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster import ledger as L  # noqa: E402
from agent.cluster.acceptance import accept_node, mock_benchmark_runner  # noqa: E402
from agent.cluster.executors import KubeExecutor, NoopExecutor  # noqa: E402
from agent.cluster.health import NodeMetrics, Verdict, evaluate_node  # noqa: E402
from agent.cluster.heal import execute_manual, plan_remediation  # noqa: E402
from agent.cluster.playbook import primary_diagnosis  # noqa: E402
from agent.cluster.provider import MockProvider, sweep  # noqa: E402

# Deterministic timestamps (no wall clock → byte-identical demo runs).
T0 = "2026-06-26T00:00:00+00:00"   # fault detected
T1 = "2026-06-26T00:03:00+00:00"   # diagnosed
T2 = "2026-06-26T00:06:00+00:00"   # auto-healed (disk)
T3 = "2026-06-26T00:41:00+00:00"   # human-resolved (drain)


def line(title: str) -> None:
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


def main() -> int:
    led = Path(tempfile.mkdtemp()) / "incidents.jsonl"
    audit = led.parent / "audit.jsonl"

    # ------------------------------------------------------------------ R2
    line("1. Bring-up acceptance — a node ships only if it clears every baseline")
    for node in ("gpu-node-000", "gpu-node-001"):  # 001 injects an HBM+NCCL regression
        res = accept_node(node, "NVIDIA H100 80GB HBM3", mock_benchmark_runner)
        verdict = "ACCEPTED" if res.accepted else "REJECTED"
        print(f"  {verdict:<9} {node}", end="")
        if res.accepted:
            print("  (all 9 checks ≥ floor → cleared for production)")
        else:
            print(f"  → rejected on: {', '.join(f.label for f in res.failures())}")

    # ------------------------------------------------------------------ R1
    line("2. Fleet 巡检 — fail-closed health verdict per node")
    fleet = sweep(MockProvider(size=6))
    for n in fleet:
        h = evaluate_node(n)
        worst = next((r.message for r in h.reasons if r.verdict != Verdict.PASS), "healthy")
        print(f"  [{h.verdict.label}] {n.node_id:<16} {worst}")

    # ------------------------------------------------------------------ R1 fault
    line("3. Inject a fault & localize it — XID 79, GPU fell off the bus")
    faulty = NodeMetrics(node_id="gpu-node-007", gpu_model="NVIDIA H100 80GB HBM3",
                         reachable=True, gpu_temp_c=58.0, ecc_uncorrectable=0,
                         nvlink_down=0, rdma_link_down=0, throttled=False,
                         xid_errors=(79,), collected_at=T0)
    h = evaluate_node(faulty)
    diag = primary_diagnosis(h)
    print(f"  verdict   : {h.verdict.label}  (signal: {h.fail_reasons()[0].signal})")
    print(f"  root cause: {diag.root_cause}")
    print(f"  remediation: {diag.remediation}")
    print(f"  risk={diag.risk}  confidence={diag.confidence}")

    # ------------------------------------------------------------------ R4 gate
    line("4. Gated remediation — auto-heal the safe case, escalate the risky one")

    # 4a: a LOW-risk disk-pressure node auto-heals (node gpu-node-001 from the fleet).
    disk_node = next(n for n in fleet if n.node_id == "gpu-node-001")
    disk_diag = primary_diagnosis(evaluate_node(disk_node))
    iid_disk = L.record_detection(node_id=disk_node.node_id, kind="disk_used_frac",
                                  severity="WARN", detected_at=T0, ledger=led)
    L.record_diagnosis(iid_disk, root_cause=disk_diag.root_cause, action=disk_diag.action,
                       remediation=disk_diag.remediation, diagnosed_at=T1, ledger=led)
    plan = plan_remediation(disk_node.node_id, disk_diag, executor=NoopExecutor(),
                            allow=True, audit_path=audit, incident_id=iid_disk, ledger=led)
    # plan_remediation recorded the auto-heal recovery; stamp it for a deterministic MTTR.
    _restamp_recovery(led, iid_disk, T2)
    print(f"  {disk_node.node_id}: decision={plan.decision.upper()}  action={plan.action}")
    print(f"            → {plan.reason}")

    # 4b: the XID-79 node is HIGH-stakes → ESCALATE; a human runs the drain.
    iid_xid = L.record_detection(node_id=faulty.node_id, kind="xid_errors",
                                 severity="FAIL", detected_at=T0, ledger=led)
    L.record_diagnosis(iid_xid, root_cause=diag.root_cause, action=diag.action,
                       remediation=diag.remediation, diagnosed_at=T1, ledger=led)
    esc = plan_remediation(faulty.node_id, diag, executor=NoopExecutor(),
                           allow=True, audit_path=audit, incident_id=iid_xid, ledger=led)
    print(f"  {faulty.node_id}: decision={esc.decision.upper()}  → human-in-the-loop")
    # Human approves and runs the drain via the Kubernetes backend (commands captured).
    kube_cmds: list = []
    kube = KubeExecutor(runner=lambda argv: (kube_cmds.append(argv) or (0, "ok")))
    execute_manual(faulty.node_id, "drain_and_reboot", kube, operator="oncall",
                   approved=True, audit_path=audit, incident_id=iid_xid, ledger=led)
    # Stamp the human recovery time for the demo.
    _restamp_recovery(led, iid_xid, T3)
    print(f"            human ran: {' '.join(kube_cmds[-1])}")

    # ------------------------------------------------------------------ R1/R4 MTTR
    line("5. Measured MTTR & self-heal ratio (the numbers are computed, not claimed)")
    stats = L.mttr_stats(led)
    print(f"  incidents      : {stats['total']}  (recovered {stats['recovered']}, "
          f"escalated {stats['escalated']})")
    print(f"  mean MTTR      : {stats['mttr_seconds_mean'] / 60:.1f} min")
    print(f"  self-heal ratio: {stats['self_heal_ratio']:.0%}  "
          f"(auto {stats['auto_healed']} / recovered {stats['recovered']})")

    print("\nThat's the loop: accept on a baseline, sweep fail-closed, localize with "
          "provenance, auto-heal only what's safe, escalate the rest — and measure MTTR.")
    return 0


def _restamp_recovery(ledger: Path, incident_id: str, ts: str) -> None:
    """Rewrite the recovery event's timestamp so the demo MTTR is deterministic."""

    import json

    lines = ledger.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        ev = json.loads(ln)
        if ev.get("incident_id") == incident_id and ev.get("event") == "recovered":
            ev["ts"] = ts
        out.append(json.dumps(ev, ensure_ascii=False))
    ledger.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
