#!/usr/bin/env python3
"""Grounding gate: close the cross-entity generalization gap at LOW false-positive cost.

`tools/run_cross_entity.py` shows the limit — structure transfers across entities
but flags every attribution (FP ≈ 1). This adds the missing piece: ground each
asserted attribution against the knowledge base, so true attributions are
recognised, contradictions are flagged, and off-KB works ABSTAIN (fail-closed).

Deterministic + offline (uses the local KB snapshot). Exits non-zero if the
grounding invariants fail.

    python tools/run_grounding_gate.py [--seed N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.cross_entity import _structural_fp, _structural_recall, entity_disjoint_split  # noqa: E402
from provenance_bench.dataset import DATA_DIR  # noqa: E402
from provenance_bench.grounded import build_kb, run_grounded  # noqa: E402


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
    return pairs, controls, true


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    pairs, controls, true = _load()
    kb = build_kb(true)
    _, test = entity_disjoint_split(pairs, seed=args.seed)
    structural_recall = _structural_recall(test)         # fast: regex over test split
    structural_fp = _structural_fp(controls)             # fast: regex over true controls
    g = run_grounded(pairs, controls, kb, seed=args.seed)

    invariants = {
        # grounding cuts the structural detector's false positives toward zero
        "grounded_cuts_false_positives": g["groundedFalsePositive"] <= 0.1,
        "grounding_beats_structural_FP": g["groundedFalsePositive"] < structural_fp,
        # and still transfers across UNSEEN entities the KB covers
        "grounded_transfers_on_covered": g["groundedRecall_covered"] >= 0.8,
        # off-KB works abstain (fail-closed), so coverage is honestly bounded
        "grounded_abstains_off_kb": g["abstainRate"] > 0.0,
    }
    ok = all(invariants.values())

    if args.json:
        base = {"structuralRecall": structural_recall, "structuralFalsePositive": structural_fp}
        print(json.dumps({"structural": base, "grounded": g, "invariants": invariants, "ok": ok}, indent=2))
        return 0 if ok else 1

    print("                         recall   false-positive")
    print(f"  structural (UNSEEN)   {structural_recall:>6.1%}     {structural_fp:>6.1%}")
    print(f"  GROUNDED (covered)    {g['groundedRecall_covered']:>6.1%}     {g['groundedFalsePositive']:>6.1%}"
          f"   (coverage {g['kbCoverage']:.0%}, abstain {g['abstainRate']:.0%})")
    print(f"\n{g['interpretation']}")
    print("\nFalsifiable invariants:")
    for name, passed in invariants.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print("\n" + ("ALL INVARIANTS HOLD" if ok else "INVARIANT FAILURE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
