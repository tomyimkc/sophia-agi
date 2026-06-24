#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the hardened SSIL orchestrator over the 12-gate hardening stack.

Builds the clean demo bundle (every gate's demo_bundle merged so the full stack
promotes), runs ``run_hardened``, prints the public-report JSON, and exits non-zero
if the aggregate verdict is not ``promote`` — the same convention as the other
run_ssil_* tools, so this composes into a CI gate.

  --reject [GATE]  instead run the scripted rejecting path (one gate forced to reject)
                   to demonstrate fail-closed worst-wins; this exits non-zero by design.

Output is candidateOnly / level3Evidence=false / canClaimAGI=false.

See docs/11-Platform/Safe-Self-Improvement-Loop.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_hardened import (  # noqa: E402
    clean_demo_bundle,
    rejecting_demo_bundle,
    run_hardened,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Hardened SSIL orchestrator (12 gates, worst-wins)")
    ap.add_argument("--candidate-id", default="sophia-rlvr-v1", help="candidate id to gate")
    ap.add_argument(
        "--reject",
        nargs="?",
        const="G8",
        default=None,
        metavar="GATE",
        help="run the scripted rejecting path forcing GATE to reject (default G8)",
    )
    ap.add_argument("--out", default=None, help="optional path to also write the report JSON")
    args = ap.parse_args()

    if args.reject is not None:
        bundle = rejecting_demo_bundle(gate_id=args.reject)
    else:
        bundle = clean_demo_bundle()

    report = run_hardened(bundle, candidate_id=args.candidate_id)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")

    # Same convention as run_ssil_*: non-zero unless the stack promoted.
    return 0 if report["verdict"] == "promote" else 1


if __name__ == "__main__":
    raise SystemExit(main())
