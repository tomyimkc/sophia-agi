#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Close the self-extending loop on a held-out domain and print the falsifiable result.

    python tools/run_selfextend_loop.py [--json]

Offline, deterministic. The domain is a NON-TRIVIAL one the real synthesis engine can
genuinely solve but a single-token stump provably cannot: "is this token a multiple of
5?" — the signal is divisibility (numeric), not a lexical token. Demonstrates: abstain
-> synthesize a COMPOSITIONAL verifier (agent.verifier_synthesis) -> meta-verify on a
disjoint split -> verified-reward selection -> anti-gaming check -> measured lift on the
held-out test split -> competence flips abstain->answer. (Selection-based; live-RL needs
a GPU.)

For comparison, the toy single-token decision stump (selfextend.verifier_synthesis) is
also scored on the same domain: it cannot express divisibility and plateaus near chance,
which is exactly why the capstone loop now routes through the real engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from selfextend import close_loop  # noqa: E402
from selfextend.verifier_synthesis import (  # noqa: E402
    stratified_split, synthesize_verifier, validate,
)


def _divisibility_domain() -> "list[tuple[str, bool]]":
    """A held-out domain whose signal is divisibility-by-5 (numeric), NOT a single
    lexical token. A single-token stump cannot express this; the real engine composes
    a `divisible_by_5` template and validates it on a disjoint split."""
    valid = [str(n) for n in range(5, 101, 5)]  # 20 multiples of 5
    invalid = ["7", "13", "22", "31", "44", "48", "52", "61", "77", "88",
               "99", "103", "ab", "3x", "12", "19"]  # none divisible by 5
    return [(v, True) for v in valid] + [(i, False) for i in invalid]


def _toy_baseline(examples: "list[tuple[str, bool]]") -> float:
    """Score the toy single-token stump on the same domain (it cannot express
    divisibility, so it plateaus near chance — the contrast that motivates the real engine)."""
    train, heldout = stratified_split(examples)
    rule = synthesize_verifier(train)
    return validate(rule, heldout) if rule else 0.0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    examples = _divisibility_domain()
    report = close_loop("divisible_by_5 (held-out)", examples)
    report["toyStumpBaselineHeldout"] = _toy_baseline(examples)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["loop_closed"] else 1

    print(f"Self-extending loop on '{report['domain']}' (real engine):")
    print(f"  verifier promoted on held-out : {report['promoted']} (acc {report['heldoutAccuracy']})")
    print(f"  synthesized gate              : {report['rule']['gate']}")
    print(f"  policy accuracy  pre -> post  : {report['preAccuracy']} -> {report['postAccuracy']} "
          f"(+{report['improvement']})")
    print(f"  anti-gaming (gate vs oracle)  : hacked={report['antiGamingCheck']['hacked']}")
    print(f"  competence route              : {report['routeBefore']} -> {report['routeAfter']}")
    print(f"  eval calibration ECE          : {report['evalCalibrationECE']}")
    print(f"  TOY single-token stump held-out: {report['toyStumpBaselineHeldout']} (near chance — "
          f"cannot express divisibility)")
    print("\n  invariants:")
    for k, v in report["invariants"].items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  LOOP CLOSED: {report['loop_closed']}")
    print(f"\n{report['interpretation']}")
    return 0 if report["loop_closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
