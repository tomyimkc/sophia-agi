#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decontamination assertion for the Sophrosyne EXTERNAL temperance battery (pillar 6).

Asserts the external-battery prompts are disjoint from every training corpus in the
repo (the battery lives under agi-proof/benchmark-results/, outside assert_decontam's
EVAL_GLOBS). Same two-layer contract via tools/_battery_decontam.

    python3 tools/assert_sophrosyne_decontam.py
    python3 tools/assert_sophrosyne_decontam.py --jaccard 0.6 --shingle 5
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

BATTERY = ROOT / "agi-proof" / "benchmark-results" / "sophrosyne" / "sophrosyne_external_battery.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--shingle", type=int, default=5)
    args = ap.parse_args()
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    prompts = [c["text"] for c in battery["cases"]]
    return assert_battery_decontam(prompts, label="SOPHROSYNE", jaccard=args.jaccard, shingle=args.shingle)


if __name__ == "__main__":
    raise SystemExit(main())
