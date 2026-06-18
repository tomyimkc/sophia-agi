#!/usr/bin/env python3
"""Export training/examples/*.json to a single JSONL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "training" / "corpus.jsonl"
EXAMPLES_DIR = ROOT / "training" / "examples"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export training examples to JSONL")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    examples = sorted(EXAMPLES_DIR.glob("*.json"))
    if not examples:
        print("No examples found in training/examples/")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_handle:
        for path in examples:
            payload = json.loads(path.read_text(encoding="utf-8"))
            out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} example(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())