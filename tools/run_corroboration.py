#!/usr/bin/env python3
"""Corroboration-aware confidence: independent agreement raises belief (the axis
min-over-chain misses). Reports the structural curve, a calibration comparison,
and falsifiable invariants. Exits non-zero if an invariant fails.

    python tools/run_corroboration.py [--seed N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.corroboration import run_demo  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    r = run_demo(seed=args.seed)
    if args.json:
        print(json.dumps(r, indent=2))
        return 0 if r["ok"] else 1

    c = r["curve"]
    print("Corroboration-aware confidence (independent agreement raises belief)")
    print("=" * 68)
    print(f"\n  1 independent source @0.7         -> {c['1src']}")
    print(f"  2 independent sources @0.7        -> {c['2src']}")
    print(f"  3 independent sources @0.7        -> {c['3src']}")
    print(f"  3 sources, SAME group (dup) @0.7  -> {c['dup3same']}   (no inflation)")
    print(f"  2 support + 1 dissent @0.2        -> {c['dissent']}   (dissent lowers)")
    print(f"\nselective risk @50% coverage:  {r['selectiveRisk']}")
    print(f"ECE (reported, not gated — noisy at this N):  {r['ece']}")
    print("\nFalsifiable invariants:")
    for name, ok in r["invariants"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\n" + ("ALL INVARIANTS HOLD" if r["ok"] else "INVARIANT FAILURE"))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
