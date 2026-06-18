#!/usr/bin/env python3
"""Bundle repo stats and leaderboards into web/data/manifest.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DATA = ROOT / "web" / "data" / "manifest.json"
DOMAINS = ("philosophy", "psychology", "history", "religion")


def main() -> int:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    examples = len(list((ROOT / "training" / "examples").glob("*.json")))
    leaderboards = {}
    for domain in DOMAINS:
        path = ROOT / "benchmark" / "results" / f"leaderboard-{domain}.json"
        if path.exists():
            leaderboards[domain] = json.loads(path.read_text(encoding="utf-8"))

    payload = {
        "version": version,
        "trainingExamples": examples,
        "domains": list(DOMAINS),
        "leaderboards": leaderboards,
        "links": {
            "github": "https://github.com/tomyimkc/sophia-agi",
            "huggingface": "https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus",
        },
    }
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {WEB_DATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())