# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""C3 driver — headline-grade PIF/SSA. --dry-run/--model mock = CI core (synthetic
cells through the shipping harness). --model <hf id> = opt-in heavy run (DEFERRED:
records the OPEN live claim; full N>=8/K>=20 on a downloaded model + held-out family)."""
from __future__ import annotations

import argparse, json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "agi-proof" / "benchmark-results" / "pif.public-report.json"


def _offline_invariants() -> "tuple[bool, dict]":
    from agent.steering import pif_harness as pif
    import random
    rng = random.Random(2); K = 24
    def cell_scores(strong):
        steer = [(1.0 if strong else 0.2) + 0.05 * rng.gauss(0, 1) for _ in range(K)]
        base = [(0.1 if strong else 0.2) + 0.05 * rng.gauss(0, 1) for _ in range(K)]
        neu = [0.0] * K
        s = {"E": {"steer": steer, "base": base, "neutral": neu}}
        for ax in ("O", "C", "A"):
            off_steer = [0.02 * rng.gauss(0, 1) for _ in range(K)]
            off_base = [0.02 * rng.gauss(0, 1) for _ in range(K)]
            off_neutral = list(off_steer)   # separate copy; cohen_d(steer, neutral)=0 by construction
            s[ax] = {"steer": off_steer, "base": off_base, "neutral": off_neutral}
        s["kappa"], s["coherence"], s["capability_drop"] = 0.6, 90.0, 0.02
        return s
    grid = [{"cell_id": "strong", "target_axis": "E", "off_target_axes": ["O", "C", "A"], "is_mock": False, "seed": 1},
            {"cell_id": "null", "target_axis": "E", "off_target_axes": ["O", "C", "A"], "is_mock": False, "seed": 2}]
    cells = pif.build_cells_from_scores({"strong": cell_scores(True), "null": cell_scores(False)}, grid)
    h = pif.headline(cells)
    checks = {"strongEnacts": cells[0]["verdict"]["status"] == "enacted",
              "nullAbstains": cells[1]["verdict"]["status"] == "abstained",
              "headlineCounts": h["total"] == 2}
    return all(checks.values()), {"checks": checks, "headline": h,
                                  "cells": [{"cell_id": c["cell_id"], "verdict": c["verdict"]["status"],
                                             "reason": c["verdict"]["reason"]} for c in cells]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.model == "mock" or args.dry_run:
        ok, detail = _offline_invariants()
        detail.update(benchmark="pif", mode="mock-offline",
                      liveClaimStatus="Open — see agi-proof/failure-ledger.md pif-headline-run-not-yet-gated-2026-06-23")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("PIF HARNESS VERIFIED ✓" if ok else "PIF HARNESS NOT MET ✗")
        return 0 if ok else 1
    # DEFERRED: gate each shipped steering vector through agent.steering.anti_gaming.ship_steering() on the held-out family before counting it.
    print("Full N>=8/K>=20 PIF run is DEFERRED (OPEN in the ledger). Build CI-green; "
          "trigger only on a non-null reduced-slice trend.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
