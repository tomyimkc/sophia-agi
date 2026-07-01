#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Focus Efficiency Frontier — the Prosoche headline benchmark harness.

Falsifiable claim (pre-registered in
``agi-proof/benchmark-results/prosoche/measurement_spec.json``): the goal-anchored
context policy solves tasks in FEWER tokens than a recency-chop / priority-packed
baseline, at EQUAL-OR-BETTER success, without losing the ability to adapt on a
legitimate goal shift.

This harness is INTENTIONALLY ``NO-GO``: the real arms need a live model/agent and
>= 2 independent judge families. What it *can* do offline and deterministically is:

  1. Run the routing battery through the Prosoche gate to report routing fidelity
     (NOT a real-decision effect), and
  2. Run the three context-packing arms on a synthetic, goal-anchored fixture and
     report the *mechanism* — how many tokens of off-goal context each arm admits —
     as ILLUSTRATIVE evidence the relevance policy holds fewer off-goal tokens.

Neither is promoted: the emitted ``focus-efficiency.PENDING.public-report.json`` is
``verdict: NO-GO`` with the missing GO requirements enumerated. Mirrors
``tools/run_sophrosyne_eval.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.context_manager import ContextManager, Segment  # noqa: E402
from agent.prosoche import (  # noqa: E402
    AttentionAnchor,
    anchor_segment,
    assess_attention,
    relevance_boost,
)

RESULTS = ROOT / "agi-proof" / "benchmark-results" / "prosoche"
BATTERY = RESULTS / "prosoche-battery.json"
REPORT = RESULTS / "focus-efficiency.PENDING.public-report.json"


def _routing_fidelity() -> dict:
    """Run the author-labelled battery through the gate — routing fidelity only."""
    spec = json.loads(BATTERY.read_text(encoding="utf-8"))
    anchor = AttentionAnchor.from_dict(spec["anchor"])
    correct = 0
    rows = []
    for item in spec["items"]:
        d = assess_attention(item["text"], anchor, context=item.get("context"))
        ok = d.verdict == item["expectVerdict"]
        correct += int(ok)
        rows.append({"id": item["id"], "got": d.verdict, "want": item["expectVerdict"], "ok": ok})
    return {
        "n": len(spec["items"]),
        "routingAccuracy": round(correct / max(1, len(spec["items"])), 4),
        "rows": rows,
        "note": "routing fidelity vs AUTHOR labels — NOT a real-decision effect, NOT >= 2-judge ground truth",
    }


def _packing_mechanism() -> dict:
    """Illustrative mechanism: off-goal token admission across the three arms.

    Deterministic, no model. Builds a window of one on-goal + several off-goal
    history segments under a tight budget and reports how many off-goal tokens each
    arm admits. The anchored arm should admit the fewest (the mechanism behind the
    efficiency claim) — reported as ILLUSTRATIVE, never as the effect.
    """
    anchor = AttentionAnchor(
        goal="diagnose the slow checkout database query",
        in_scope_entities=("checkout", "database", "query", "index"),
    )
    on_goal = Segment(kind="prior", text=("The checkout database query does a full table scan on orders; "
                                          "adding an index on customer_id would cut the query latency. " * 4),
                      priority=5, provenance="on-goal")
    off_goal = [
        Segment(kind="prior", text=("Unrelated office trivia about the coffee machine rota and lunch menu. " * 4),
                priority=6, provenance=f"off-goal-{i}")
        for i in range(3)
    ]
    segs_baseline = [on_goal, *off_goal]                       # priority-packed: off-goal ranks high (newer)
    segs_anchored = [anchor_segment(anchor), on_goal, *off_goal]
    budget = 140

    def off_goal_tokens(res) -> int:
        cm = ContextManager(budget)
        return sum(cm.counter(s.text) for s in res.segments if s.provenance.startswith("off-goal"))

    priority_arm = ContextManager(budget).pack(segs_baseline)
    anchored_arm = ContextManager(budget, relevance_fn=relevance_boost(anchor)).pack(segs_anchored)

    return {
        "budgetTokens": budget,
        "priorityPackedOffGoalTokens": off_goal_tokens(priority_arm),
        "prosocheAnchoredOffGoalTokens": off_goal_tokens(anchored_arm),
        "anchorIsCacheStable": anchored_arm.stable_prefix_tokens > 0,
        "note": "ILLUSTRATIVE mechanism only (no model, no judge); fewer off-goal tokens != a measured token-per-solved-task effect",
    }


def build_report() -> dict:
    return {
        "experimentId": "focus-efficiency-frontier",
        "headline": "PENDING — mechanism + routing only; no real model/agent token-per-solved-task run has been performed",
        "verdict": "NO-GO",
        "go": False,
        "status": "not_run",
        "battery": "sophia.prosoche_battery.v1",
        "harness": "tools/run_focus_efficiency_frontier.py",
        "preregistration": "agi-proof/benchmark-results/prosoche/measurement_spec.json",
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "delta": None,
        "routingFidelity": _routing_fidelity(),
        "packingMechanism": _packing_mechanism(),
        "arms": {
            "recency-chop-baseline": {"status": "not_run", "reason": "requires a real model/agent"},
            "priority-packed-baseline": {"status": "not_run", "reason": "requires a real model/agent"},
            "prosoche-anchored": {"status": "not_run", "reason": "scored only alongside real baselines + judges"},
        },
        "criticalFailures": [
            "no_real_arms: tokens-per-solved-task needs a live model/agent (mechanism counts are not the effect)",
            "ground_truth_not_2family: solved/on-goal labels are author-only, not >= 2 independent judge families (kappa >= 0.40)",
            "no_effect_ci: delta tokens-per-solved-task CI does not exclude 0 (no real arms to compute it)",
            "no_task_success_guardrail: the success guardrail (delta success >= -0.02) needs a real task run",
            "no_antifixation_guardrail: the goal-shift subset success-floor needs a real task run",
        ],
        "note": ("Intentionally PENDING. Promotion needs an external decontaminated, drift-balanced task set, "
                 ">= 2 independent judge families, both baseline arms, a delta tokens-per-solved-task CI excluding 0, "
                 "the task-success guardrail, the anti-fixation goal-shift guardrail, and the safety floor "
                 "(no safety step pruned as off-goal). See the measurement_spec and the prosoche-attention-gate row "
                 "in agi-proof/failure-ledger.md."),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Focus Efficiency Frontier harness (NO-GO by design).")
    ap.add_argument("--write", action="store_true", help="write the PENDING public report artifact")
    ap.add_argument("--check", action="store_true", help="verify the committed report matches a fresh build")
    args = ap.parse_args()

    report = build_report()
    if args.check:
        if not REPORT.exists():
            print("MISSING report artifact", file=sys.stderr)
            return 2
        on_disk = json.loads(REPORT.read_text(encoding="utf-8"))
        # Compare only the stable, deterministic skeleton (verdict + critical failures
        # + routing accuracy), not the full nested counts, to stay drift-robust.
        same = (on_disk.get("verdict") == report["verdict"]
                and on_disk.get("criticalFailures") == report["criticalFailures"]
                and on_disk.get("routingFidelity", {}).get("routingAccuracy")
                == report["routingFidelity"]["routingAccuracy"])
        print("OK" if same else "DRIFT")
        return 0 if same else 3

    out = json.dumps(report, indent=2, sort_keys=True)
    if args.write:
        REPORT.write_text(out + "\n", encoding="utf-8")
        print(f"wrote {REPORT.relative_to(ROOT)}  verdict={report['verdict']}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
