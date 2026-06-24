#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cross-entity generalization: do learned provenance rules transfer to UNSEEN entities?

Falsifiable contrast, on an entity-disjoint train/test split:
  - memorized rules    -> high recall on seen entities, ~0 on unseen (no transfer),
                          ~0 false positives;
  - structural detector-> high recall on unseen entities (transfers) BUT ~1 false
                          positives (cannot tell true from false attributions).

The honest conclusion: low-false-positive cross-entity generalization requires
external grounding, not pattern memorization. Exits non-zero if invariants fail.

    python tools/run_cross_entity.py [--seed N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.cross_entity import run_cross_entity  # noqa: E402
from provenance_bench.dataset import DATA_DIR  # noqa: E402


def _load() -> tuple:
    mis = json.loads((DATA_DIR / "misattributions.json").read_text(encoding="utf-8"))["misattributions"]
    pairs = [{"claimed": m["claimed_author"], "work": m["work"]} for m in mis]
    true = json.loads((DATA_DIR / "wikidata_snapshot.json").read_text(encoding="utf-8"))["attributions"]
    controls = [
        {"gold": t["gold_author"], "work": t["work"]}
        for t in true
        if "(" not in t["gold_author"] and "anonymous" not in t["gold_author"].lower()
        and " and " not in t["gold_author"]
    ]
    return pairs, controls


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    pairs, controls = _load()
    r = run_cross_entity(pairs, controls, seed=args.seed)

    invariants = {
        "split_is_entity_disjoint": r["entityDisjoint"],
        "memorized_works_within_entity": r["withinEntityRecall"] >= 0.8,
        "memorized_does_NOT_transfer_across_entities": r["crossEntityRecall_memorized"] <= 0.1,
        "structural_DOES_transfer_across_entities": r["crossEntityRecall_structural"] >= 0.8,
        "structural_is_imprecise_high_FP": r["structuralFalsePositive"] >= 0.5,
        "memorized_is_precise_zero_FP": r["memorizedFalsePositive"] == 0.0,
    }
    ok = all(invariants.values())

    if args.json:
        print(json.dumps({**r, "invariants": invariants, "ok": ok}, indent=2))
        return 0 if ok else 1

    print(f"entity-disjoint split: train={r['nTrain']} test={r['nTest']} disjoint={r['entityDisjoint']}")
    print("\n                         recall   false-positive")
    print(f"  memorized (seen)      {r['withinEntityRecall']:>6.1%}        —")
    print(f"  memorized (UNSEEN)    {r['crossEntityRecall_memorized']:>6.1%}     {r['memorizedFalsePositive']:>6.1%}")
    print(f"  structural (UNSEEN)   {r['crossEntityRecall_structural']:>6.1%}     {r['structuralFalsePositive']:>6.1%}")
    print(f"\n{r['interpretation']}")
    print("\nFalsifiable invariants:")
    for name, passed in invariants.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("\n" + ("ALL INVARIANTS HOLD" if ok else "INVARIANT FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
