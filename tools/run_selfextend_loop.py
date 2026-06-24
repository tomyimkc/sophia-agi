#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Close the self-extending loop on a held-out domain and print the falsifiable result.

    python tools/run_selfextend_loop.py [--json]

Offline, deterministic. Demonstrates: abstain -> synthesize verifier -> validate ->
verified-reward selection -> measured improvement on an independent eval split ->
competence flips abstain->answer. (Selection-based; live-RL needs a GPU.)
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


def _danger_domain(n: int = 12) -> "list[tuple[str, bool]]":
    """A held-out domain with a stable signal token ('delete') across all splits."""
    objs = ["the database", "user files", "records", "everything", "the backups",
            "the logs", "all accounts", "the cache", "the index", "the config",
            "the queue", "the secrets"]
    pos = [(f"delete {o} now", True) for o in objs[:n]]
    neg = [(f"read {o} now", False) for o in objs[:n]]
    # interleave so order doesn't bias the stratified split
    out: list = []
    for p, q in zip(pos, neg):
        out += [p, q]
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    report = close_loop("danger_intent (held-out)", _danger_domain())
    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["loop_closed"] else 1

    print(f"Self-extending loop on '{report['domain']}':")
    print(f"  verifier promoted on held-out : {report['promoted']} (acc {report['heldoutAccuracy']})")
    print(f"  policy accuracy  pre -> post  : {report['preAccuracy']} -> {report['postAccuracy']} "
          f"(+{report['improvement']})")
    print(f"  generalizes on eval split     : {report['postAccuracy']} >= threshold")
    print(f"  competence route              : {report['routeBefore']} -> {report['routeAfter']}")
    print(f"  eval calibration ECE          : {report['evalCalibrationECE']}")
    print("\n  invariants:")
    for k, v in report["invariants"].items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  LOOP CLOSED: {report['loop_closed']}")
    print(f"\n{report['interpretation']}")
    return 0 if report["loop_closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
