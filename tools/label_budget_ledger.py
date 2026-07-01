#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Oracle-label budget ledger — fail-closed to ADVISORY when supply is exhausted.

The blind spot: gates that lean on a SCARCE label source (human graders, a metered
proprietary judge, a private held-out gold set that gets consumed as it is spent)
cannot keep blocking once that source runs dry. The tempting failure is to let such
a gate SILENTLY PASS on empty/stale labels. That is worse than no gate. The honest
move is to DEMOTE the gate to 'advisory' (non-blocking, makes no claim) and say so
out loud.

This module reads/updates ``agi-proof/label-budget.json`` (per-oracle labels
available / spent / which gates draw on it) and answers two questions:

  * ``spend(oracle, n)`` — record that a run consumed ``n`` labels from an oracle
    (clamped so spent never exceeds available; over-request is reported).
  * ``status()`` — per-oracle remaining, and the set of gates DEMOTED to advisory
    because at least one oracle they depend on is exhausted (remaining <= 0).

A gate is demoted if ANY oracle it depends on is exhausted (conjunctive: a gate is
only as blocking as its scarcest input). Exit code is ALWAYS 0 — this is a ledger,
not a pass/fail gate — but demotions are returned in the receipt for the caller to
act on (a demoted gate must be treated as non-blocking by whatever consumes this).

    python3 tools/label_budget_ledger.py                              # report status
    python3 tools/label_budget_ledger.py --spend metered_judge_openrouter:1500
    python3 tools/label_budget_ledger.py --spend heldout_gold_set:40 --write
    python3 tools/label_budget_ledger.py --json

Without ``--write`` the ledger file is NOT modified (dry-run spend for preview).
JSON receipt to stdout; human prose to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "agi-proof" / "label-budget.json"


def load_ledger(path: Path) -> dict:
    """Load the label-budget ledger."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_ledger(ledger: dict, path: Path) -> None:
    """Persist the ledger back to disk (called only under --write)."""
    Path(path).write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def spend(ledger: dict, oracle: str, n: int) -> dict:
    """Record that ``n`` labels were drawn from ``oracle``.

    ``labelsSpent`` is clamped to ``labelsAvailable`` (you cannot spend labels you
    do not have); the amount requested beyond the budget is reported as ``overRequested``
    so a caller cannot quietly overdraw. Mutates ``ledger`` in place.
    """
    oracles = ledger.get("oracles", {})
    if oracle not in oracles:
        raise KeyError(f"unknown oracle: {oracle!r}")
    o = oracles[oracle]
    avail = int(o.get("labelsAvailable", 0))
    spent = int(o.get("labelsSpent", 0))
    requested = spent + int(n)
    new_spent = min(requested, avail)
    over = max(0, requested - avail)
    o["labelsSpent"] = new_spent
    return {"oracle": oracle, "spentNow": int(n), "totalSpent": new_spent,
            "available": avail, "overRequested": over}


def status(ledger: dict) -> dict:
    """Compute per-oracle remaining and the gates demoted to advisory.

    A gate is DEMOTED (fail-closed to non-blocking) when any oracle it depends on
    is exhausted (remaining <= 0).
    """
    oracles = ledger.get("oracles", {})
    per_oracle: dict[str, dict] = {}
    exhausted_oracles: list[str] = []
    gate_to_oracles: dict[str, list[str]] = {}

    for name, o in oracles.items():
        avail = int(o.get("labelsAvailable", 0))
        spent = int(o.get("labelsSpent", 0))
        remaining = avail - spent
        is_exhausted = remaining <= 0
        per_oracle[name] = {
            "available": avail,
            "spent": spent,
            "remaining": remaining,
            "exhausted": is_exhausted,
            "gates": list(o.get("gates", [])),
        }
        if is_exhausted:
            exhausted_oracles.append(name)
        for g in o.get("gates", []):
            gate_to_oracles.setdefault(g, []).append(name)

    demoted: list[dict] = []
    active: list[str] = []
    for gate, deps in sorted(gate_to_oracles.items()):
        dead = [d for d in deps if per_oracle[d]["exhausted"]]
        if dead:
            demoted.append({"gate": gate, "status": "advisory",
                            "reason": "oracle supply exhausted",
                            "exhaustedOracles": dead, "dependsOn": deps})
        else:
            active.append(gate)

    return {
        "tool": "label_budget_ledger",
        "status": "preregistration_only",
        "canClaimAGI": False,
        "perOracle": per_oracle,
        "exhaustedOracles": exhausted_oracles,
        "gatesActive": sorted(active),
        "gatesDemoted": demoted,
        "anyDemotions": bool(demoted),
    }


def _parse_spend(spec: str) -> tuple[str, int]:
    """Parse an ``oracle:count`` spend spec."""
    if ":" not in spec:
        raise ValueError(f"--spend expects oracle:count, got {spec!r}")
    oracle, _, cnt = spec.rpartition(":")
    return oracle, int(cnt)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Oracle-label budget ledger; demotes starved gates to advisory.")
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="path to label-budget.json")
    ap.add_argument("--spend", default=None, help="record a draw as oracle:count (e.g. metered_judge_openrouter:1500)")
    ap.add_argument("--write", action="store_true", help="persist the spend to the ledger file (default: dry-run)")
    ap.add_argument("--json", action="store_true", help="print only the JSON receipt")
    args = ap.parse_args(argv)

    try:
        ledger = load_ledger(Path(args.ledger))
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"[label_budget_ledger] cannot read ledger: {e}", file=sys.stderr)
        return 2

    spend_result = None
    if args.spend:
        try:
            oracle, n = _parse_spend(args.spend)
            spend_result = spend(ledger, oracle, n)
        except (ValueError, KeyError) as e:
            print(f"[label_budget_ledger] bad --spend: {e}", file=sys.stderr)
            return 2
        if args.write:
            try:
                save_ledger(ledger, Path(args.ledger))
            except OSError as e:
                print(f"[label_budget_ledger] cannot write ledger: {e}", file=sys.stderr)
                return 2

    receipt = status(ledger)
    if spend_result is not None:
        receipt["spend"] = spend_result
        receipt["written"] = bool(args.write)

    print(json.dumps(receipt, indent=2, ensure_ascii=False))

    if not args.json:
        if receipt["gatesDemoted"]:
            names = ", ".join(d["gate"] for d in receipt["gatesDemoted"])
            print(f"[label_budget_ledger] {len(receipt['gatesDemoted'])} gate(s) DEMOTED to advisory "
                  f"(oracle exhausted): {names}", file=sys.stderr)
        else:
            print("[label_budget_ledger] all gates active; no oracle exhausted", file=sys.stderr)
        if spend_result is not None:
            tag = "written" if args.write else "dry-run (use --write to persist)"
            over = f", OVER by {spend_result['overRequested']}" if spend_result["overRequested"] else ""
            print(f"[label_budget_ledger] spend {spend_result['oracle']} += {spend_result['spentNow']} "
                  f"(total {spend_result['totalSpent']}/{spend_result['available']}{over}) [{tag}]", file=sys.stderr)
    # Always 0: this is a ledger, not a pass/fail gate. Demotions are in the receipt.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
