#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fleet inspection sweep (R1: 巡检 / fault detection + localization).

Collect telemetry for every node, compute a fail-closed health verdict, and (with
``--diagnose``) attach a root-cause hypothesis + proposed remediation per fault. With
``--ledger`` it opens an incident for every WARN/FAIL node so MTTR can be measured.

    # offline, deterministic synthetic fleet (default, no network, no cost)
    python3 tools/cluster/inspect_fleet.py
    python3 tools/cluster/inspect_fleet.py --diagnose --json
    # live RunPod inventory (needs RUNPOD_API_KEY)
    python3 tools/cluster/inspect_fleet.py --source runpod --diagnose --ledger
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
from agent.cluster.health import Verdict, evaluate_node  # noqa: E402
from agent.cluster.playbook import diagnose, primary_diagnosis  # noqa: E402
from agent.cluster.provider import get_provider, sweep  # noqa: E402


def run_inspection(source: str, *, size: int, want_diagnose: bool,
                   inventory: str | None = None, ssh_key: str | None = None) -> list[dict]:
    provider = get_provider(source, size=size, inventory=inventory, ssh_key=ssh_key)
    out: list[dict] = []
    for metrics in sweep(provider):
        health = evaluate_node(metrics)
        rec = health.to_dict()
        if want_diagnose and health.verdict != Verdict.PASS:
            rec["diagnoses"] = [d.to_dict() for d in diagnose(health)]
            primary = primary_diagnosis(health)
            rec["primary_diagnosis"] = primary.to_dict() if primary else None
        out.append(rec)
    return out


def _print_table(records: list[dict]) -> None:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for rec in records:
        counts[rec["verdict"]] = counts.get(rec["verdict"], 0) + 1
        node = rec["node_id"]
        verdict = rec["verdict"]
        top = ""
        bad = [r for r in rec["reasons"] if r["verdict"] != "PASS"]
        if bad:
            top = bad[0]["message"]
        print(f"  [{verdict}] {node:<16} {top}")
        if rec.get("primary_diagnosis"):
            d = rec["primary_diagnosis"]
            print(f"         ↳ {d['root_cause']}")
            print(f"           fix: {d['remediation']}  (risk={d['risk']}, conf={d['confidence']})")
    total = len(records)
    print(f"\n  fleet: {total} nodes — PASS {counts['PASS']} · WARN {counts['WARN']} · FAIL {counts['FAIL']}")


def _record_ledger(records: list[dict], ledger: Path) -> int:
    opened = 0
    for rec in records:
        if rec["verdict"] == "PASS":
            continue
        worst = max(rec["reasons"], key=lambda r: {"PASS": 0, "WARN": 1, "FAIL": 2}[r["verdict"]])
        iid = ledger_mod.record_detection(
            node_id=rec["node_id"], kind=worst["signal"], severity=rec["verdict"],
            ledger=ledger,
        )
        primary = rec.get("primary_diagnosis")
        if primary:
            ledger_mod.record_diagnosis(
                iid, root_cause=primary["root_cause"], action=primary["action"],
                remediation=primary["remediation"], ledger=ledger,
            )
        opened += 1
    return opened


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sophia cluster inspection sweep (R1).")
    ap.add_argument("--source", default="mock", choices=["mock", "runpod", "ssh"],
                    help="telemetry source (default: mock, offline)")
    ap.add_argument("--size", type=int, default=6, help="synthetic fleet size (mock only)")
    ap.add_argument("--inventory", default=None,
                    help="JSON node inventory for --source ssh (or SOPHIA_CLUSTER_INVENTORY)")
    ap.add_argument("--ssh-key", default=None,
                    help="SSH private key path for --source ssh (or SOPHIA_CLUSTER_SSH_KEY)")
    ap.add_argument("--diagnose", action="store_true", help="attach root-cause diagnoses")
    ap.add_argument("--ledger", action="store_true", help="open incidents for WARN/FAIL nodes")
    ap.add_argument("--ledger-path", default=str(ledger_mod.DEFAULT_LEDGER))
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args(argv)

    try:
        records = run_inspection(args.source, size=args.size, want_diagnose=args.diagnose,
                                 inventory=args.inventory, ssh_key=args.ssh_key)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    opened = 0
    if args.ledger:
        opened = _record_ledger(records, Path(args.ledger_path))

    if args.json:
        print(json.dumps({"nodes": records, "incidents_opened": opened}, ensure_ascii=False, indent=2))
    else:
        _print_table(records)
        if args.ledger:
            print(f"  opened {opened} incident(s) → {args.ledger_path}")

    # Exit nonzero if any node FAILed, so CI / cron can alert on it.
    return 1 if any(r["verdict"] == "FAIL" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
