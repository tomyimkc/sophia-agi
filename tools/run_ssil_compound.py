#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the compounding SSIL loop and emit an artifact.

Offline (default): a scripted proposer drives a deterministic compounding curve —
promote -> replicate -> canonical -> beat-the-canonical -> converge — with the full
two-key gate set (G1 value + G3 capability + G2/G4/G5/G6) on every round.

Live (--live, needs a provider key): a real model proposes each round and is probed
on corrigibility/honeypots. Output: candidateOnly / canClaimAGI=false.

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

from agent.ssil_compound import demo_compound_report, run_compound_loop, scripted_proposer  # noqa: E402
from agent.ssil_proposer import propose_policy_spec  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-extension" / "ssil-compound.public-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compounding SSIL loop")
    ap.add_argument("--live", action="store_true", help="use a live model proposer + probes")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--canonical-n", type=int, default=2)
    ap.add_argument("--spec", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--registry", default=None, help="optional append-only registry path")
    args = ap.parse_args()

    if args.live:
        def live_proposer(round_idx, baseline):
            policy, _ = propose_policy_spec(spec=args.spec)
            return {k: policy[k] for k in ("min_sources", "min_quality", "default_action")}
        rep = run_compound_loop(live_proposer, rounds=args.rounds, canonical_n=args.canonical_n,
                                seed=7, registry_path=args.registry, live_probes=True, model_spec=args.spec)
        rep.setdefault("invariants", {})
    else:
        rep = demo_compound_report()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rounds={rep['rounds']} curve={rep['compoundingCurve']} "
          f"finalBest={rep['finalCanonicalBest']['spec'] if rep['finalCanonicalBest'] else None}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
