#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the canary circuit-breaker against an auto-approver (AATS experiment 4 driver).

Loads a labelled canary set and runs it through a chosen approver, then persists and
reports the breaker state. This is the safety gate that ships BEFORE autonomy: if the
approver false-approves a planted bad item (or false-rejects a planted good one) the
breaker trips and this tool EXITS NON-ZERO, so any pipeline that gates autonomy on it
reverts to human-only.

Two reference approvers (offline, deterministic, no model):

  * ``--approver sound`` — an AND-panel of the repo's real deterministic verifiers
    (arithmetic + temporal-date + provenance-corpus) over controlled data; it approves
    the known-good probes and rejects the known-bad ones, so the breaker stays armed.
  * ``--approver leaky`` — approves everything (a stand-in for a miscalibrated/gamed
    auto-approver); it approves a planted bad item, so the breaker TRIPS.

    python tools/run_canary_breaker.py --approver sound     # stays armed, exit 0
    python tools/run_canary_breaker.py --approver leaky      # trips, exit 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.auto_approval_breaker import CircuitBreaker, load_canary_set  # noqa: E402

CANARY_SET = ROOT / "agi-proof" / "aats" / "canary-set.jsonl"
STATE_PATH = ROOT / "agi-proof" / "aats" / "breaker-state.json"
REPORT_PATH = ROOT / "agi-proof" / "aats" / "canary-breaker.public-report.json"


def sound_approver():
    """AND-panel of real deterministic verifiers over controlled data (offline)."""
    from agent.temporal_verifier import temporal_consistent
    from agent.verifiers import arithmetic_sound, provenance_faithful

    arith = arithmetic_sound()
    temporal = temporal_consistent({
        "authors": {"Aristotle": {"died": -322}},
        "works": {"Critique of Pure Reason": {"created": 1781}, "Hamlet": {"created": 1600}},
    })
    provenance = provenance_faithful({
        "critique_of_pure_reason": {"canonicalTitleEn": "Critique of Pure Reason",
                                    "doNotAttributeTo": ["Aristotle"]},
    })

    def approve(text: str) -> bool:
        return all(v(text, None, {})["passed"] for v in (arith, temporal, provenance))

    return approve


def leaky_approver():
    """A miscalibrated approver that rubber-stamps everything."""
    return lambda _text: True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run canary circuit-breaker (AATS exp 4).")
    ap.add_argument("--approver", choices=["sound", "leaky"], default="sound")
    ap.add_argument("--canaries", type=Path, default=CANARY_SET)
    ap.add_argument("--state", type=Path, default=STATE_PATH)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--reset", metavar="OPERATOR",
                    help="re-arm a tripped breaker as OPERATOR (logged), then exit")
    args = ap.parse_args(argv)

    breaker = CircuitBreaker.load(args.state)

    if args.reset:
        breaker.reset(operator=args.reset, reason="manual re-arm via run_canary_breaker --reset")
        breaker.save(args.state)
        print(f"breaker re-armed by {args.reset}")
        return 0

    approver = sound_approver() if args.approver == "sound" else leaky_approver()
    canaries = load_canary_set(args.canaries)
    report = breaker.check_canaries(approver, canaries)
    report["approver"] = args.approver
    breaker.save(args.state)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"canary breaker (approver={args.approver}, {report['nCanaries']} canaries)")
    for r in report["results"]:
        flag = "ok" if r["ok"] else "MISS"
        print(f"  [{flag}] {r['id']:28s} expectApprove={r['expectApprove']!s:5s} approved={r['approved']}")
    print(f"  breaker: {report['breaker'].upper()}"
          + (f" — {report['trippedReason']}" if report["trippedReason"] else ""))
    print(f"Wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    # exit non-zero if tripped so autonomy gates revert to human-only
    return 1 if breaker.tripped else 0


if __name__ == "__main__":
    raise SystemExit(main())
