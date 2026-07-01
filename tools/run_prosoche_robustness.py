#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prosoche robustness probe — explicit anchor vs DERIVED goal (the honest gap).

Mirror of the Andreia / Sophrosyne / Dikaiosyne robustness probes. The Prosoche
gate is strongest when the attention anchor is GIVEN (explicit goal + in-scope
entities). In real use the goal must often be DERIVED from raw conversation, with
no entity scope. This probe measures the cost of that, on a small author battery:

  * EXPLICIT mode  — route each item against its full anchor (goal + entities).
  * DERIVED mode   — route each item against only a vague paraphrase of the goal,
                     with NO in-scope entities (entity drift degrades to N/A, so the
                     decision rests on the weaker lexical semantic signal alone).

The finding is reported, NOT tuned away: explicit routing far exceeds derived
routing. This is a fail-closed property (the gate cannot flag drift from a goal it
cannot read), and the fix is model-gated (a learned goal-extractor / semantic
backend behind the default-off seam) — see the
`prosoche-derived-goal-weak-vs-explicit-anchor` failure-ledger row. canClaimAGI:false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prosoche import AttentionAnchor, assess_attention  # noqa: E402

OUT = ROOT / "agi-proof" / "benchmark-results" / "prosoche" / "prosoche-robustness.json"

# Each item: a full anchor, a vague derived-goal paraphrase (no entities), the step
# text, and the expected verdict. Single-axis, author-written (routing fidelity only).
BATTERY = [
    {
        "id": "login-focus",
        "goal": "fix the failing auth login test in services.auth",
        "entities": ["services.auth", "login", "auth test"],
        "derivedGoal": "sort out the broken test thing",
        "text": "Looking at the failing login test in services.auth: the auth token check rejects valid sessions.",
        "expect": "focused",
    },
    {
        "id": "login-drift",
        "goal": "fix the failing auth login test in services.auth",
        "entities": ["services.auth", "login", "auth test"],
        "derivedGoal": "sort out the broken test thing",
        "text": "While I'm here, let me rewrite the unrelated Marketing Page and recolour the Telemetry Dashboard.",
        "expect": "drifting",
    },
    {
        "id": "checkout-focus",
        "goal": "optimise the slow checkout database query latency",
        "entities": ["checkout", "database", "query", "latency"],
        "derivedGoal": "make the thing faster",
        "text": "The checkout database query does a full table scan; an index on customer_id cuts the query latency.",
        "expect": "focused",
    },
    {
        "id": "checkout-drift",
        "goal": "optimise the slow checkout database query latency",
        "entities": ["checkout", "database", "query", "latency"],
        "derivedGoal": "make the thing faster",
        "text": "Let me instead reorganise the office snack inventory and the meeting-room booking calendar.",
        "expect": "drifting",
    },
]


def _route(anchor: AttentionAnchor, text: str) -> str:
    return assess_attention(text, anchor).verdict


def run() -> dict:
    rows = []
    exp_correct = der_correct = 0
    for it in BATTERY:
        explicit = AttentionAnchor(goal=it["goal"], in_scope_entities=tuple(it["entities"]))
        derived = AttentionAnchor(goal=it["derivedGoal"])  # no entities, vague goal
        ev = _route(explicit, it["text"])
        dv = _route(derived, it["text"])
        exp_ok = ev == it["expect"]
        der_ok = dv == it["expect"]
        exp_correct += int(exp_ok)
        der_correct += int(der_ok)
        rows.append({"id": it["id"], "expect": it["expect"],
                     "explicitVerdict": ev, "explicitOk": exp_ok,
                     "derivedVerdict": dv, "derivedOk": der_ok})
    n = len(BATTERY)
    explicit_acc = round(exp_correct / n, 4)
    derived_acc = round(der_correct / n, 4)
    return {
        "schema": "sophia.prosoche_robustness.v1",
        "n": n,
        "explicitAccuracy": explicit_acc,
        "derivedAccuracy": derived_acc,
        "gap": round(explicit_acc - derived_acc, 4),
        "rows": rows,
        "candidateOnly": True,
        "canClaimAGI": False,
        "finding": (
            f"Explicit-anchor routing {explicit_acc} vs derived-goal routing {derived_acc} "
            f"(gap {round(explicit_acc - derived_acc, 4)}). The derived signal is weaker BY DESIGN "
            "(fail-closed: the gate cannot flag drift from a goal it cannot read); the fix is "
            "model-gated behind the default-off goal-extraction seam, not a threshold relaxation."
        ),
        "boundary": (
            "Routing fidelity vs author labels only — NOT a real-decision effect. No claim that the "
            "gate detects drift from raw text without an explicit anchor. canClaimAGI:false."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the robustness artifact")
    ap.add_argument("--check", action="store_true", help="verify the committed artifact matches a fresh run")
    args = ap.parse_args()

    report = run()
    if args.check:
        if not OUT.exists():
            print("MISSING robustness artifact", file=sys.stderr)
            return 2
        on_disk = json.loads(OUT.read_text(encoding="utf-8"))
        same = (on_disk.get("explicitAccuracy") == report["explicitAccuracy"]
                and on_disk.get("derivedAccuracy") == report["derivedAccuracy"])
        print("OK" if same else "DRIFT")
        return 0 if same else 3

    out = json.dumps(report, indent=2, sort_keys=True)
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(out + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}  explicit={report['explicitAccuracy']} derived={report['derivedAccuracy']}",
              file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
