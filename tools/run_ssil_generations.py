#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Emit the generational compounding proof artifact (the rising gated canonical curve
+ negative control). The state machine is deterministic; per-generation 3-seed
aggregates come from real RunPod merge-then-train runs (or the demo fixtures).

  --gens-json '<json>'   inline list of generations, each:
        {"gen":1,"adapterId":"sophia-rlvr-v1","trainedOn":"Qwen2.5-3B",
         "runs":[{"seed":0,"after":0.70,"before":0.53,"protected_before":0.79,
                  "protected_after":0.79,"contaminated":false}, ...]}
  (no arg) -> demo fixtures.

Exits non-zero unless the proof shows compounding under the gate (>=2 monotone
generations) AND the negative control diverges. canClaimAGI=false.

See docs/11-Platform/Safe-Self-Improvement-Loop.md (compounding on weights).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_aggregate import AdapterAggregate, SeedRun  # noqa: E402
from agent.ssil_generations import Generation, compounding_proof, demo_compounding_report  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-extension" / "ssil-compounding-proof.public-report.json"


def _gens_from_json(rows: list[dict]) -> list[Generation]:
    gens = []
    for row in rows:
        runs = [SeedRun(seed=int(r["seed"]), before=float(r["before"]), after=float(r["after"]),
                        protected_before=float(r["protected_before"]), protected_after=float(r["protected_after"]),
                        contaminated=bool(r.get("contaminated", False))) for r in row["runs"]]
        agg = AdapterAggregate(adapter_id=row["adapterId"], config={"adapter": row["adapterId"], "kind": "lora_adapter"},
                               runs=runs, canonical_n=int(row.get("canonicalN", 3)))
        gens.append(Generation(gen=int(row["gen"]), adapter_id=row["adapterId"],
                               trained_on=row.get("trainedOn", "?"), aggregate=agg))
    return gens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gens-json", default=None)
    ap.add_argument("--min-delta", type=float, default=0.03)
    ap.add_argument("--ci-k", type=float, default=1.0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args(argv)

    if args.gens_json:
        proof = compounding_proof(_gens_from_json(json.loads(args.gens_json)), min_delta=args.min_delta, ci_k=args.ci_k)
        proof["invariants"] = {
            "compounds_under_gate": proof["proves"]["compounds_under_gate"],
            "negative_control_diverges": proof["proves"]["negative_control_diverges"],
        }
    else:
        proof = demo_compounding_report()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")

    g = proof["gated"]
    ok = proof["proves"]["compounds_under_gate"] and proof["proves"]["negative_control_diverges"]
    if args.print:
        print(json.dumps(proof, ensure_ascii=False, indent=2))
    print(f"gated curve={g['curve']} monotone={g['monotoneRising']} convergedAt={g['convergedAt']} "
          f"gateCaught={proof['gateCaughtGenerations']} gateMattered={proof['gateMadeADifference']}")
    print(f"compounding proof: {'PASS' if ok else 'INCOMPLETE'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
