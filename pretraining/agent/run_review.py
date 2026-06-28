#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the pretraining-researcher reviewer agent over the study artifacts.

    python -m pretraining.agent.run_review            # deterministic, offline
    python -m pretraining.agent.run_review --llm      # + optional additive LLM critique
    python -m pretraining.agent.run_review --out review.json

Exit code is 0 when there are no unassessed studies (every report exists), 1 otherwise —
so this doubles as a regression check that the studies still produce their artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pretraining.agent.researcher import review_all

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llm", action="store_true", help="add an optional LLM critique")
    ap.add_argument("--out", type=Path, default=HERE / "review-latest.json")
    args = ap.parse_args()

    review = review_all(llm=args.llm)
    args.out.write_text(json.dumps(review, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    t = review["tally"]
    print(f"overall: {review['overall']}  "
          f"(pass={t['pass']} concern={t['concern']} cannot_assess={t['cannot_assess']})")
    for name, s in review["studies"].items():
        flag = {"pass": "PASS", "concern": "CONC", "cannot_assess": "????"}[s["verdict"]]
        print(f"  [{flag}] {name:18s} {s['evidence']}")
        for c in s["critiques"]:
            print(f"         ! {c}")
    if review.get("llm_critique", {}).get("available"):
        print("\nLLM critique:\n" + review["llm_critique"]["text"])
    sys.exit(1 if t["cannot_assess"] else 0)


if __name__ == "__main__":
    main()
