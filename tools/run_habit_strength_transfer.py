#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Habit-Strength Transfer (HST) harness — H5 identity-consistency + power analysis.

Pre-registration: agi-proof/benchmark-results/habit-formation/measurement_spec.json
Design note:      docs/06-Roadmap/Atomic-Habits-for-Sophia.md

WHAT THIS DOES (offline, deterministic, no GPU):
  1. Scores identity-consistency (agent.identity_consistency, H5) on the EXISTING
     M3-transfer answer pack for the base and adapter columns — 160 novel entities.
  2. Computes the paired in-character vote diff, a bootstrap CI, an anytime-valid
     confidence sequence, and a McNemar exact test (tools.eval_stats).
  3. Runs the power analysis the spec leaves to-be-computed: mde_at_n(N) and the
     required N for the observed effect — the power-before-you-run pillar.
  4. Emits a candidate receipt.

WHAT THIS IS NOT. This is the BASE-vs-ADAPTER comparison on an existing pack — it
validates the harness and gives an *illustrative* identity-consistency delta. It is
**not** the flagship HST claim (flat-reward arm vs graded-craving arm, >=3 seeds, >=2
judge families), which requires GPU training of two adapters and must go through the
gate before any promotion. Receipt is ``candidateOnly: true``, ``validated: false``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.identity_consistency import identity_consistency, paired_vote_diffs
from tools.eval_stats import (
    bootstrap_ci_paired,
    confidence_sequence_mean,
    mcnemar,
    mde_at_n,
    required_n_for_mde,
)

DEFAULT_PACK = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "M3-transfer-answers.json"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "habit-formation" / "hst-identity-consistency-existing-pack.json"


def run(pack_path: Path, base_key: str, adapter_key: str) -> dict:
    cases = json.loads(pack_path.read_text())
    if not isinstance(cases, list):
        raise SystemExit(f"expected a list of cases in {pack_path}")
    n = len(cases)

    base = identity_consistency(cases, base_key)
    adapter = identity_consistency(cases, adapter_key)

    diffs = paired_vote_diffs(cases, base_key, adapter_key)  # adapter - base, in {-1,0,1}
    delta = round(sum(diffs) / n, 4) if n else None
    ci = bootstrap_ci_paired(diffs, seed=0)
    cs = confidence_sequence_mean(diffs)                      # anytime-valid (spec: peeked metric)
    mc = mcnemar([int(not v["fabricated"]) for v in base["perCase"]],
                 [int(not v["fabricated"]) for v in adapter["perCase"]])

    # Power-before-you-run. paired_rho=0 is the conservative default the claim gate uses;
    # same-items pairing means the true rho>0, so this OVER-states the required N (honest).
    mde = round(mde_at_n(n), 4) if n else None
    req_n = required_n_for_mde(abs(delta)) if delta else None

    ci_excludes_zero = bool(ci[0] is not None and (ci[0] > 0 or ci[1] < 0))
    underpowered = bool(mde is not None and abs(delta or 0.0) < mde)

    return {
        "experimentId": "habit-formation-hst-v1",
        "artifact": "hst-identity-consistency-existing-pack",
        "candidateOnly": True,
        "validated": False,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false; harness validation + illustrative delta",
        "comparison": "BASE vs ADAPTER on the existing M3-transfer pack (NOT flat-vs-graded reward arms)",
        "pack": str(pack_path.relative_to(ROOT)),
        "n": n,
        "metric": "identity-consistency (H5) = fraction of cases with no forbidden assertion committed",
        "baseRate": base["rate"],
        "adapterRate": adapter["rate"],
        "baseRouteAppropriateRate": base["routeAppropriateRate"],
        "adapterRouteAppropriateRate": adapter["routeAppropriateRate"],
        "delta": delta,
        "bootstrapCI95": ci,
        "anytimeValidCS95": cs,
        "mcnemar": mc,
        "power": {
            "mde_at_n": mde,
            "requiredNForObservedEffect": req_n,
            "paired_rho": 0.0,
            "note": "conservative paired_rho=0; same-items pairing means true rho>0, so requiredN is overstated",
        },
        "ciExcludesZero": ci_excludes_zero,
        "underpowered": underpowered,
        "honestLimits": [
            "single existing answer pack (one set of answers per case); not >=3 seeds",
            "deterministic forbidden-assertion markers only; no LLM-judge family here (the spec requires >=2 for any VALIDATED claim)",
            "base-vs-adapter, NOT the pre-registered flat-vs-graded reward arms — that needs GPU training",
        ],
        "nextStep": "train flat-reward vs graded-craving (H2) adapters, >=3 seeds, judge identity-consistency with >=2 families, then claim_gate --prefix HST --assert-prereg",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="HST identity-consistency harness (offline)")
    ap.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    ap.add_argument("--base-key", default="base_answer")
    ap.add_argument("--adapter-key", default="adapter_answer")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--print-only", action="store_true", help="do not write the receipt")
    args = ap.parse_args()

    report = run(args.pack, args.base_key, args.adapter_key)
    # Drop the verbose per-case arrays from the persisted receipt (keep it auditable but compact).
    summary = {k: v for k, v in report.items()}
    print(json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, indent=2))
    if not args.print_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print(f"\nreceipt -> {args.out.relative_to(ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
