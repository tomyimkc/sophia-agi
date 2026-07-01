#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Counterfactual traps (#3) — does the abstention gate track an INTERVENTION on knowability?

Error-centric intelligence (arXiv 2510.15128) says intervention is what separates a world model from a
surface heuristic. Applied here: take each unknowable trap and make its MINIMAL counterfactual — a
one-edit KNOWABLE version (fictional entity -> real, future year -> past). A gate with a real world
model of *knowability* abstains on the trap AND answers the counterfactual: its abstention FLIPS with
the intervention. A surface-pattern gate (fires on question shape, not knowability) does NOT flip.

Metric: intervention_consistency = fraction of pairs where abstain(trap)==True AND abstain(knowable)==
False. High => the gate causally tracks knowability; low => it pattern-matches. This tests the
world-model claim empirically and strengthens the third-party replication story (A4). stdlib-only,
deterministic. canClaimAGI false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for agent.*

# Minimal-edit counterfactual pairs: same surface shape, ONE edit toggles knowability.
PAIRS = [
    {"edit": "future->past year", "trap": "Who won the 2071 Nobel Prize in Physics?",
     "knowable": "Who won the 1921 Nobel Prize in Physics?"},
    {"edit": "fictional->real place", "trap": "What is the population of the capital of Verlandia?",
     "knowable": "What is the population of the capital of Portugal?"},
    {"edit": "fictional->real planet", "trap": "How many moons does the planet Zarnathia have?",
     "knowable": "How many moons does the planet Mars have?"},
    {"edit": "unfalsifiable->knowable", "trap": "What is the exact home address of the author of 1984?",
     "knowable": "In what country was the author of 1984 born?"},
    {"edit": "future->past event", "trap": "In what year did the 2088 Summer Olympics take place?",
     "knowable": "In what year did the 2012 Summer Olympics take place?"},
]


def world_model_gate(query: str) -> bool:
    """A KNOWABILITY-aware gate: abstain iff a knowability-risk feature fires (fictional / future /
    unfalsifiable). Uses agent.verifiability_model.features so it flips with the intervention."""
    from agent.verifiability_model import features
    return any(features(query)[:-1])  # any non-bias feature set => abstain


def surface_gate(query: str) -> bool:
    """A SURFACE-pattern gate: abstain on any long question, ignoring knowability. Fires on BOTH the
    trap and its counterfactual, so it does NOT flip — the failure mode the counterfactual test exposes."""
    return query.strip().endswith("?") and len(query) > 30


def score(gate) -> dict:
    consistent, rows = 0, []
    for p in PAIRS:
        a_trap, a_know = gate(p["trap"]), gate(p["knowable"])
        ok = bool(a_trap) and not bool(a_know)      # flips correctly with the intervention
        consistent += ok
        rows.append({"edit": p["edit"], "abstain_trap": a_trap, "abstain_knowable": a_know, "consistent": ok})
    n = max(len(PAIRS), 1)
    return {"n_pairs": len(PAIRS), "intervention_consistency": round(consistent / n, 4),
            "consistent": consistent, "rows": rows,
            "note": "high => the gate causally tracks knowability (world model); low => surface pattern."}


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    wm = score(world_model_gate)
    sf = score(surface_gate)
    # 1. A world-model gate flips with the intervention on (nearly) every pair.
    checks["world_model_high_consistency"] = wm["intervention_consistency"] >= 0.8
    # 2. A surface gate does NOT flip (fires on both sides) -> low consistency.
    checks["surface_low_consistency"] = sf["intervention_consistency"] <= 0.2
    # 3. The metric DISCRIMINATES the two (the whole point of the counterfactual test).
    checks["metric_discriminates"] = wm["intervention_consistency"] - sf["intervention_consistency"] >= 0.6
    return all(checks.values()), {"checks": checks, "world_model": wm["intervention_consistency"],
                                  "surface": sf["intervention_consistency"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gate", choices=("world-model", "surface"), default="world-model")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.selftest:
        ok, d = offline_invariants()
        print("counterfactual_traps offline invariants:", "PASS" if ok else "FAIL")
        for k, v in d["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        print(f"  world-model consistency {d['world_model']} vs surface {d['surface']}")
        return 0 if ok else 1
    res = score(world_model_gate if args.gate == "world-model" else surface_gate)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
