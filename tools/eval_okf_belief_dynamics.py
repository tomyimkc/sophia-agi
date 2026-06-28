#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Belief-dynamics evidence driver for the OKF forgetting layer (CANDIDATE).

The OKF is append-only today: a belief's ``authorConfidence`` is static once written.
``okf.decay_okf`` / ``okf.frontier_demotion`` / ``okf.forgetting_audit`` add a dynamics
layer (Bayesian-ish decay, surprise-gating, competition suppression, a decisive-evidence
rule for frontier consensus, and a tamper-evident audit trail) WITHOUT weakening source
discipline. This driver exercises that layer on its falsifiable cases and emits an
auditable report — converting the unit tests into a reproducible evidence artifact.

It runs three evidence panels, each with a deterministic pass/fail the report names:

  1. HONESTY PROPERTIES of decay (P1-P3):
       P1 no-silent-deletion — forgetting is demotion, never destruction.
       P2 provenanced-forgetting — every suppression carries an auditable reason.
       P3 source-discipline-outranks-recency — consensus is never time-decayed.

  2. HISTORICAL SIMULATIONS of frontier demotion (the thesis-level falsifiable cases):
       - Newton -> Einstein: decisive evidence (K>=100, N>=3 independent, all
         surprise-gated) -> regime-scoped demotion, exactly ONE rank. PASS.
       - OPERA faster-than-light neutrino: one high-surprise event, N=1 < floor ->
         QUARANTINE, consensus untouched (the rule that protected special relativity). PASS.

  3. AUDIT TRAIL tamper-evidence — the hash-chained ledger verifies clean, and a single
     bit-flip breaks the chain (non-repudiation of forgetting decisions).

Honest bound: this is deterministic, offline, pure-stdlib dynamics over *supplied* inputs.
It does not compute Bayes factors from raw data and does not touch model weights; the
decisions are audited candidate decisions, not validated learning. ``level3Evidence: false``
until a real run wires this into the consolidation loop and clears the anti-forgetting gate.
Reproduce: ``python tools/eval_okf_belief_dynamics.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "agi-proof" / "okf-consistency" / "belief-dynamics.public-report.json"

NOW = 1_700_000_000.0  # fixed epoch so the report is byte-reproducible


# --------------------------------------------------------------------------
# Panel 1 — honesty properties of decay
# --------------------------------------------------------------------------

def _belief(node_id, conf, age_days, surprise=0.0, reinforced=0):
    from okf.decay_okf import BeliefState
    return BeliefState(
        node_id=node_id, author_confidence=conf,
        written_at=NOW - age_days * 86400, last_reinforced_at=NOW - age_days * 86400,
        surprise=surprise, reinforcement_count=reinforced, decayed_reason=None,
    )


def _panel_decay_honesty() -> dict:
    """Exercise P1-P3 directly against plan_decay."""
    from okf.decay_okf import plan_decay  # local import keeps the panel self-describing

    # P1: over many beliefs, no deletion is ever emitted.
    p1_beliefs = [_belief(f"n{i}", "attributed", age_days=i * 400) for i in range(8)]
    p1_plan = plan_decay(p1_beliefs, now=NOW)
    p1_pass = p1_plan.deletions == 0 and p1_plan.to_dict()["noSilentDeletion"] is True

    # P2: every suppression carries a controlled-vocabulary reason.
    p2_beliefs = [
        _belief("old_weak", "legendary", age_days=2000),   # very decayed -> suppress
        _belief("fresh", "attributed", age_days=10),
    ]
    p2_plan = plan_decay(p2_beliefs, now=NOW)
    p2_reasons = {n: r for n, r in p2_plan.suppress}
    p2_pass = (
        "old_weak" in p2_reasons
        and p2_reasons["old_weak"].split(":", 1)[0] in {"time", "contradiction", "competition"}
    )

    # P3: a consensus belief, however old and unreinforced, is NOT suppressed by time.
    p3_plan = plan_decay([_belief("c", "consensus", age_days=50_000)], now=NOW)
    p3_pass = "c" not in {n for n, _ in p3_plan.suppress}

    return {
        "panel": "decay-honesty",
        "properties": {
            "P1_noSilentDeletion": {"pass": p1_pass, "deletions": p1_plan.deletions},
            "P2_provenancedForgetting": {
                "pass": p2_pass,
                "suppressedReason": p2_reasons.get("old_weak"),
            },
            "P3_consensusImmuneToTime": {"pass": p3_pass},
        },
        "pass": p1_pass and p2_pass and p3_pass,
    }


# --------------------------------------------------------------------------
# Panel 2 — historical simulations of frontier consensus demotion
# --------------------------------------------------------------------------

def _panel_frontier_demotion() -> dict:
    """The two paradigm cases: a correct demotion and a correct non-demotion."""
    from okf.frontier_demotion import simulate_newton_to_einstein, simulate_opera_ftl_neutrino

    einstein = simulate_newton_to_einstein()
    opera = simulate_opera_ftl_neutrino()

    einstein_pass = (
        einstein["demoted"]
        and einstein["regimeScoped"]
        and einstein["newConfidence"] == "disputed"          # exactly ONE rank
        and einstein["stillConsensusInLowVelocity"]
    )
    opera_pass = (
        opera["quarantined"]
        and opera["consensusUntouched"]
        and any("multiplicity floor" in r for r in opera["reason"])
    )

    return {
        "panel": "frontier-demotion",
        "simulations": {
            "newtonToEinstein": {
                "pass": einstein_pass,
                "claim": "decisive evidence -> regime-scoped demotion, ONE rank",
                **{k: v for k, v in einstein.items() if k != "decision"},
            },
            "operaFtlNeutrino": {
                "pass": opera_pass,
                "claim": "N=1 high-surprise event -> quarantine, consensus untouched",
                "quarantined": opera["quarantined"],
                "consensusUntouched": opera["consensusUntouched"],
            },
        },
        "pass": einstein_pass and opera_pass,
    }


# --------------------------------------------------------------------------
# Panel 3 — audit-trail tamper-evidence
# --------------------------------------------------------------------------

def _panel_audit_tamper_evidence() -> dict:
    """The hash-chained ledger verifies clean and detects a single bit-flip."""
    from okf.forgetting_audit import ForgettingAudit

    a = ForgettingAudit()
    a.record_plan({"suppress": [("n1", "time"), ("n2", "competition:q")],
                   "reinforce": ["n3"], "quarantine": []})
    # also fold a frontier-demotion decision into the same chain
    a.record_demotion({"demote": True, "newConfidence": "disputed",
                       "supersededByRegime": "relativistic_strong_field",
                       "rankDrop": 1, "nodeId": "newton"})
    clean_verifies = a.verify()

    a.tamper(1)  # mutate one middle record
    tamper_detected = not a.verify()

    return {
        "panel": "audit-tamper-evidence",
        "cleanChainVerifies": clean_verifies,
        "singleBitFlipDetected": tamper_detected,
        "recordsChained": len(a.to_list()),
        "pass": clean_verifies and tamper_detected,
    }


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

def run() -> dict:
    p1 = _panel_decay_honesty()
    p2 = _panel_frontier_demotion()
    p3 = _panel_audit_tamper_evidence()
    all_pass = p1["pass"] and p2["pass"] and p3["pass"]
    return {
        "schema": "sophia.okf_belief_dynamics_report.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "claimBoundary": (
            "Deterministic, offline dynamics over the OKF graph — decay, surprise-gating, "
            "competition suppression, decisive-evidence frontier demotion, and a "
            "tamper-evident audit trail. NOT a learning rule, NOT weight changes, NOT a "
            "capability claim. The forgetting layer is candidate-only and not wired into the "
            "live consolidation loop until a real run clears the anti-forgetting gate."
        ),
        "pass": all_pass,
        "panels": [p1, p2, p3],
        "honestBound": (
            "Pure-stdlib dynamics over supplied/audited inputs; Bayes factors are taken as "
            "audited input, not computed from raw data. Self-authored falsifiable cases."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OKF belief-dynamics evidence (candidate)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    report = run()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("OKF belief dynamics — candidate evidence")
        for panel in report["panels"]:
            tag = "PASS" if panel["pass"] else "FAIL"
            print(f"  [{tag}] {panel['panel']}")
            for sim in panel.get("simulations", {}).values():
                stag = "PASS" if sim["pass"] else "FAIL"
                print(f"        [{stag}] {sim['claim']}")
        print(f"  overall: {'PASS' if report['pass'] else 'FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.json:
        print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
