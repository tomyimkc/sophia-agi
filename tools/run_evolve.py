# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CLI: run one Evolve round on an evolvable target from the experience log.

Loads a target's verifier/gate-scored experiences, splits them, proposes a
verifier candidate, and canary-gates it against the current baseline — promoting
ONLY on a held-out improvement (a regression is blocked, never shipped).

    python -m tools.run_evolve --target "verifier:math"

Offline and deterministic. Prints the canary decision as JSON.
"""
from __future__ import annotations

import argparse
import json

from selfextend.experience_log import labelled_examples
from selfextend.verifier_synthesis import stratified_split, synthesize_verifier
from selfextend.evolve import evolve_verifier


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Run one Evolve canary round on a target")
    ap.add_argument("--target", required=True, help='e.g. "verifier:math"')
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="regression_eps: promote only if improvement exceeds this")
    args = ap.parse_args(argv)

    examples = labelled_examples(args.target)
    if len(examples) < 2:
        print(json.dumps({"target": args.target, "decision": "hold",
                          "reason": f"insufficient experience ({len(examples)} examples)"}, indent=2))
        return 0

    train, heldout = stratified_split(examples, frac=0.5)
    # Current baseline = a verifier synthesised from train only (the incumbent).
    baseline = synthesize_verifier(train)
    report = evolve_verifier(args.target, train, heldout, baseline=baseline,
                             regression_eps=args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
