#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the embryogenesis crucible arena (candidate infrastructure).

Population-based verifier configuration search — 8–32 embryos, no weight updates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.embryogenesis.arena import run_arena  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "embryogenesis-crucible" / "crucible-arena.public-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Embryogenesis crucible arena pilot")
    ap.add_argument("--population", type=int, default=8)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--generality-limit", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed (default 0)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run_arena(
        population_size=args.population,
        generations=args.generations,
        top_k=args.top_k,
        generality_limit=args.generality_limit,
        seed=args.seed,
    )
    report["ok"] = bool(report.get("history"))
    report["invariants"] = {
        "weights_frozen": report.get("weightsFrozen") is True,
        "population_in_bounds": 1 <= args.population <= 32,
        "has_winners": bool(report.get("winners")),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "out": str(out),
                "topFitness": report["history"][-1]["topFitness"] if report["history"] else 0,
            },
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
