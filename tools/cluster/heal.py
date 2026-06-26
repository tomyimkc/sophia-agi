#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gated auto-remediation driver (R4: 自动化运维 / 提升故障自愈率).

Inspect the fleet, diagnose every fault, and route each through the fail-closed
remediation gate: LOW-risk + high-confidence + auto-safe action ⇒ auto-heal; anything
else ⇒ escalate to a human. Every decision is audited and reflected in the incident
ledger so the self-heal ratio is measured.

Dry-run by default — NOTHING touches a node unless ``--apply`` is passed AND
``SOPHIA_CLUSTER_HEAL=1`` is set AND a real executor is wired. This CLI ships only the
no-op executor, so it always plans and records without acting.

    python3 tools/cluster/heal.py                 # plan + audit, no action
    python3 tools/cluster/heal.py --ledger --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster import ledger as ledger_mod  # noqa: E402
from agent.cluster.executors import get_executor  # noqa: E402
from agent.cluster.health import Verdict, evaluate_node  # noqa: E402
from agent.cluster.heal import Decision, plan_remediation  # noqa: E402
from agent.cluster.playbook import primary_diagnosis  # noqa: E402
from agent.cluster.provider import get_provider, sweep  # noqa: E402


def _build_executor(args):
    """Construct the remediation executor for the chosen backend."""

    if args.backend == "ssh":
        targets = {}
        if args.inventory:
            from agent.cluster.ssh_provider import SSHProvider
            for t in SSHProvider.from_inventory(args.inventory, key_path=args.ssh_key).targets:
                targets[t.node_id] = t
        return get_executor("ssh", targets=targets, key_path=args.ssh_key)
    return get_executor(args.backend)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sophia gated auto-remediation (R4).")
    ap.add_argument("--source", default="mock", choices=["mock", "runpod", "ssh"])
    ap.add_argument("--size", type=int, default=6)
    ap.add_argument("--inventory", default=None, help="JSON node inventory for --source ssh")
    ap.add_argument("--ssh-key", default=None, help="SSH private key path for --source ssh")
    ap.add_argument("--ledger", action="store_true", help="open + update incidents")
    ap.add_argument("--ledger-path", default=str(ledger_mod.DEFAULT_LEDGER))
    ap.add_argument("--apply", action="store_true",
                    help="permit real actions (still needs SOPHIA_CLUSTER_HEAL=1)")
    ap.add_argument("--backend", default="noop", choices=["noop", "ssh", "kube", "slurm"],
                    help="remediation executor backend (default: noop = simulate)")
    ap.add_argument("--node", default=None,
                    help="manual mode: execute --action on this single node (human-approved)")
    ap.add_argument("--action", default=None,
                    help="manual mode: the action to execute (e.g. drain_and_reboot, cordon_and_investigate)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    ledger_path = Path(args.ledger_path)

    executor = _build_executor(args)

    # --- Manual mode: a human runs one approved (often ESCALATED) action ---
    if args.node and args.action:
        from agent.cluster.heal import execute_manual

        ok = execute_manual(args.node, args.action, executor,
                            approved=args.apply or None, ledger=ledger_path)
        print(f"manual remediation {args.node} / {args.action}: "
              f"{'executed' if ok else 'NOT executed (need --apply + SOPHIA_CLUSTER_HEAL=1, or unsupported action)'}")
        return 0 if ok else 1

    plans: list[dict] = []
    try:
        provider = get_provider(args.source, size=args.size,
                                inventory=args.inventory, ssh_key=args.ssh_key)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for metrics in sweep(provider):
        health = evaluate_node(metrics)
        if health.verdict == Verdict.PASS:
            continue
        diag = primary_diagnosis(health)
        if diag is None:
            continue
        incident_id = None
        if args.ledger:
            worst = max(health.reasons, key=lambda r: r.verdict)
            incident_id = ledger_mod.record_detection(
                node_id=health.node_id, kind=worst.signal,
                severity=health.verdict.label, ledger=ledger_path,
            )
            ledger_mod.record_diagnosis(
                incident_id, root_cause=diag.root_cause, action=diag.action,
                remediation=diag.remediation, ledger=ledger_path,
            )
        plan = plan_remediation(
            health.node_id, diag,
            executor=executor,                 # auto-heals only run if AUTO_HEAL + approved
            allow=args.apply or None,          # None → fall back to env flag
            incident_id=incident_id, ledger=ledger_path,
        )
        plans.append(plan.to_dict())

    auto = sum(1 for p in plans if p["decision"] == Decision.AUTO_HEAL)
    esc = sum(1 for p in plans if p["decision"] == Decision.ESCALATE)
    obs = sum(1 for p in plans if p["decision"] == Decision.OBSERVE)

    if args.json:
        print(json.dumps({"plans": plans, "auto_heal": auto, "escalate": esc, "observe": obs},
                         ensure_ascii=False, indent=2))
    else:
        for p in plans:
            print(f"  [{p['decision'].upper():<9}] {p['node_id']:<16} {p['action']:<26} {p['reason']}")
        print(f"\n  {len(plans)} fault(s): auto-heal {auto} · escalate {esc} · observe {obs}"
              f"  (dry-run={'no' if args.apply else 'yes'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
