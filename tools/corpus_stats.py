#!/usr/bin/env python3
"""Print corpus statistics for README badges and release notes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def count_json_records(path: Path) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, dict) else 0


def main() -> int:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    examples = len(list((ROOT / "training" / "examples").glob("*.json")))
    bench_cases = 0
    bench_path = ROOT / "tests" / "attribution_bench.json"
    if bench_path.exists():
        bench_cases = len(json.loads(bench_path.read_text(encoding="utf-8")).get("cases", []))

    domains = json.loads((ROOT / "data" / "domains.json").read_text(encoding="utf-8"))
    active = sum(1 for d in domains.values() if d.get("status") == "active")
    planned = sum(1 for d in domains.values() if d.get("status") == "planned")

    print(f"version={version}")
    print(f"training_examples={examples}")
    print(f"philosophy_attributions={count_json_records(ROOT / 'data' / 'attributions.json')}")
    print(f"benchmark_cases={bench_cases}")
    print(f"domains_active={active}")
    print(f"domains_planned={planned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())