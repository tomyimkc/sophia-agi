#!/usr/bin/env python3
"""Export training/examples/*.json to a single JSONL corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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

    from agent.training_safety import filter_examples  # LoRA leakage guard (#7)

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in examples]
    filtered = filter_examples(payloads)   # drop confidential / PII / secret examples

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_handle:
        for payload in filtered["safe"]:
            out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    note = f" (dropped {filtered['nDropped']} unsafe: {filtered['reasonsHistogram']})" if filtered["nDropped"] else ""
    print(f"Wrote {filtered['nSafe']} safe example(s) to {args.out}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())