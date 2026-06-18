#!/usr/bin/env python3
"""Build per-domain reference response files from training examples + case_map."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "benchmark" / "reference" / "case_map.json"
EXAMPLES = ROOT / "training" / "examples"
OUT_DIR = ROOT / "benchmark" / "reference"


def assistant_text(example_path: Path) -> str:
    data = json.loads(example_path.read_text(encoding="utf-8"))
    return next(m["content"] for m in data["messages"] if m.get("role") == "assistant")


def main() -> int:
    case_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain, cases in case_map.items():
        responses = {}
        for case_id, example_file in cases.items():
            responses[case_id] = assistant_text(EXAMPLES / example_file)

        payload = {
            "domain": domain,
            "model": "sophia-teacher-reference",
            "date": "2026-06-18",
            "responses": responses,
        }
        out = OUT_DIR / f"responses-{domain}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out} ({len(responses)} responses)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())