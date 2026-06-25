#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build and print the eval coverage matrix; write it to JSON.

    python -m pretraining.eval_matrix.run_matrix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.eval_matrix.matrix import build_matrix

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "eval-matrix-latest.json")
    args = ap.parse_args()
    m = build_matrix()
    args.out.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"packs={m['n_packs']} cases={m['total_cases']} "
          f"coverage={m['covered_cells']}/{m['total_cells']} "
          f"({m['coverage_fraction']*100:.0f}%)")
    print("covered cells (dimension|domain -> cases):")
    for cell, info in sorted(m["covered"].items()):
        print(f"  {cell:28s} cases={info['cases']:5d} "
              f"auto={info['automatic']} human={info['human_or_judge']}")
    print(f"uncovered cells: {len(m['uncovered_cells'])} "
          f"(e.g. {', '.join(m['uncovered_cells'][:6])} ...)")


if __name__ == "__main__":
    main()
