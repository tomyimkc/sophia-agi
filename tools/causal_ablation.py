#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Interventional causal-ablation (#2) — discover which features CAUSALLY drive the gate.

Error-centric intelligence (arXiv 2510.15128): intervention, not observation, reveals causal structure.
Upgrade the observational autoresearch loop into an INTERVENTIONAL one: systematically inject/remove a
knowability feature into neutral base questions and measure the change in the gate's abstain-rate. The
causal effect of a feature = P(abstain | feature injected) - P(abstain | not). This maps the causal
graph of *when and why* the gate abstains — the failure-manifold analog of locate_wrong_step. stdlib.
canClaimAGI false; CANDIDATE research prototype.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STEMS = [
    "What is the population of the capital city",
    "Who won the Nobel Prize in Physics",
    "How many moons does the planet have",
    "What was the closing stock price of the company",
    "In what year did the Summer Olympics take place",
]

# Each intervention edits a neutral stem to INJECT one knowability-risk feature (or a null control).
INTERVENTIONS = {
    "inject_future_year": lambda s: f"{s} in the year 2091?",
    "inject_fictional_entity": lambda s: f"{s} of the fictional country Verlandia?",
    "inject_unfalsifiable": lambda s: f"State the exact, precise {s.lower()}?",
    "null_padding": lambda s: f"{s}, please?",             # adds text, no knowability change
}


def _gate(query: str) -> bool:
    from agent.verifiability_model import features
    return any(features(query)[:-1])


def causal_effects() -> dict:
    base_rate = sum(_gate(f"{s}?") for s in STEMS) / len(STEMS)
    effects = {}
    for name, fn in INTERVENTIONS.items():
        rate = sum(_gate(fn(s)) for s in STEMS) / len(STEMS)
        effects[name] = round(rate - base_rate, 4)
    ranked = sorted(effects.items(), key=lambda kv: -abs(kv[1]))
    return {"base_abstain_rate": round(base_rate, 4), "causal_effects": effects,
            "ranked": ranked,
            "note": "effect = P(abstain|feature injected) - P(abstain|base). High |effect| => causal "
                    "driver of abstention; ~0 => not causal (e.g. null padding)."}


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    r = causal_effects()
    e = r["causal_effects"]
    # 1. Injecting a genuine knowability-risk feature strongly increases abstention.
    checks["future_year_is_causal"] = e["inject_future_year"] >= 0.8
    checks["fictional_is_causal"] = e["inject_fictional_entity"] >= 0.8
    checks["unfalsifiable_is_causal"] = e["inject_unfalsifiable"] >= 0.8
    # 2. A null intervention (adds text, no knowability change) has ~0 causal effect.
    checks["null_is_not_causal"] = abs(e["null_padding"]) <= 0.05
    # 3. The ranking surfaces the real drivers ABOVE the null (causal discovery works).
    checks["ranking_puts_null_last"] = r["ranked"][-1][0] == "null_padding"
    return all(checks.values()), {"checks": checks, "effects": e}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        ok, d = offline_invariants()
        print("causal_ablation offline invariants:", "PASS" if ok else "FAIL")
        for k, v in d["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        print(f"  effects: {d['effects']}")
        return 0 if ok else 1
    print(json.dumps(causal_effects(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
