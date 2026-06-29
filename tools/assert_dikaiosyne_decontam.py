#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decontamination assertion for the Dikaiosyne EXTERNAL justice battery (pillar 6).

Asserts every MEMBER prompt (base + irrelevant + relevant variants, across all
equivalence classes) is disjoint from every training corpus in the repo. Same
two-layer contract via tools/_battery_decontam.

    python3 tools/assert_dikaiosyne_decontam.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._battery_decontam import assert_battery_decontam  # noqa: E402

BATTERY = ROOT / "agi-proof" / "benchmark-results" / "dikaiosyne" / "dikaiosyne_external_battery.json"


def _all_member_texts(battery: dict) -> list[str]:
    texts: list[str] = []
    for c in battery["classes"]:
        texts.append(c["base"]["text"])
        texts += [m["text"] for m in c["irrelevantVariants"]]
        texts += [m["text"] for m in c["relevantVariants"]]
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--shingle", type=int, default=5)
    args = ap.parse_args()
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    return assert_battery_decontam(_all_member_texts(battery), label="DIKAIOSYNE",
                                   jaccard=args.jaccard, shingle=args.shingle)


if __name__ == "__main__":
    raise SystemExit(main())
