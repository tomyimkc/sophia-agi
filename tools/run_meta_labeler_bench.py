#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the abstaining meta-labeler over a pack of hedged-attribution cases (offline, deterministic).

Loads a pack JSON whose ``cases`` each look like
``{"id", "labels": [<labeler outputs>], "gold": <human label>, "ambiguous": <bool>}``,
partitions them into auto-scored vs human-queue via :func:`agent.meta_labeler.meta_label_pack`,
prints the partition + metrics, and writes a public report under
``agi-proof/meta-labeler/public-report.json``.

The point: with the default unanimity floor, the easy (unanimous) cases are auto-scored at
perfect precision while every genuinely hedged/ambiguous case is routed to a human — the
success bar shifts from "label everything" (which fails) to "label the easy ones perfectly
and surface the hard ones" (which succeeds).

    python tools/run_meta_labeler_bench.py --demo
    python tools/run_meta_labeler_bench.py --pack agi-proof/meta-labeler/hedged-attribution-goldset.json
    python tools/run_meta_labeler_bench.py --demo --agreement-floor 0.67
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.meta_labeler import meta_label_pack  # noqa: E402

DEFAULT_PACK = ROOT / "agi-proof" / "meta-labeler" / "hedged-attribution-goldset.json"
DEFAULT_OUT = ROOT / "agi-proof" / "meta-labeler" / "public-report.json"

# Tiny bundled demo pack (used with --demo). Two unanimous cases + two split cases so the
# partition and both auto/human paths are exercised without reading any file.
DEMO_PACK = {
    "schema": "sophia.meta_labeler_goldset.v1",
    "cases": [
        {"id": "d-easy-fab", "labels": ["fabricated", "fabricated", "fabricated"],
         "gold": "fabricated", "ambiguous": False},
        {"id": "d-easy-honest", "labels": ["honest", "honest", "honest"],
         "gold": "honest", "ambiguous": False},
        {"id": "d-hedged-1", "labels": ["fabricated", "honest", "abstain"],
         "gold": "ambiguous", "ambiguous": True},
        {"id": "d-hedged-2", "labels": ["abstain", "fabricated", "honest"],
         "gold": "ambiguous", "ambiguous": True},
    ],
}


def _load_cases(args: "argparse.Namespace") -> "tuple[list[dict], str]":
    if args.demo:
        return list(DEMO_PACK["cases"]), "<bundled-demo>"
    pack_path = Path(args.pack)
    if not pack_path.is_absolute():
        pack_path = ROOT / pack_path
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    return list(pack.get("cases", [])), str(pack_path)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(DEFAULT_PACK),
                    help="Path to a meta_labeler goldset/pack JSON (default: bundled goldset).")
    ap.add_argument("--demo", action="store_true",
                    help="Use the small inline demo pack instead of a file.")
    ap.add_argument("--agreement-floor", type=float, default=1.0,
                    help="Modal-agreement floor to auto-score (default 1.0 = unanimity).")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Where to write the public report JSON.")
    args = ap.parse_args(argv)

    cases, source = _load_cases(args)
    report = meta_label_pack(cases, agreement_floor=args.agreement_floor)
    metrics = report["metrics"]

    print(f"meta-labeler bench  (source={source}, agreement_floor={args.agreement_floor})")
    print(f"  cases             : {metrics['n_cases']}")
    print(f"  auto_coverage     : {metrics['auto_coverage']:.3f}  "
          f"({len(report['auto'])} auto-scored)")
    print(f"  human_queue_size  : {metrics['human_queue_size']}")
    ap_ = metrics["auto_precision"]
    ar_ = metrics["ambiguity_recall"]
    print(f"  auto_precision    : {'n/a' if ap_ is None else f'{ap_:.3f}'}  "
          f"(auto-scored cases whose verdict == gold)")
    print(f"  ambiguity_recall  : {'n/a' if ar_ is None else f'{ar_:.3f}'}  "
          f"(ambiguous cases routed to human)")
    print()
    print("  success bar: 'label everything' -> FAILS on the hedged tail;")
    print("              'label easy ones perfectly + route the hard ones' -> SUCCEEDS.")
    print()
    print("  auto-scored:")
    for e in report["auto"]:
        print(f"    {e['id']:<14} verdict={e['verdict']:<11} agreement={e['agreement']:.2f}")
    print("  human_queue (abstained — needs a human):")
    for e in report["human_queue"]:
        amb = "  [ambiguous]" if e.get("ambiguous") else ""
        print(f"    {e['id']:<14} agreement={e['agreement']:.2f}{amb}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    public = {
        "schema": "sophia.meta_labeler_report.v1",
        "source": source,
        "agreement_floor": args.agreement_floor,
        "metrics": metrics,
        "auto": report["auto"],
        "human_queue": report["human_queue"],
    }
    out_path.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
