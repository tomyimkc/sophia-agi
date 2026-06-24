#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the out-of-wiki fact-check gate on text.

Offline by default. Optional retrieval backends can be wired later; this CLI is a
stable JSON front door for tests and pipelines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_gate import decision_to_dict, fact_check_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("text", nargs="?", help="text to check; stdin used if omitted")
    ap.add_argument("--json", action="store_true", help="print JSON (default true; kept for convention)")
    args = ap.parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    result = decision_to_dict(fact_check_text(text))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["verdict"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
