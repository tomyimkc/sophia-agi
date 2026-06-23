#!/usr/bin/env python3
"""Run Sophia AGI per-domain benchmark workflows."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ("philosophy", "psychology", "history", "religion", "personality")
BENCH_DIR = ROOT / "tests"
TEMPLATE_DIR = ROOT / "benchmark" / "templates"
RESULTS_DIR = ROOT / "benchmark" / "results"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_templates() -> None:
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        bench = load_json(BENCH_DIR / f"benchmark-{domain}.json")
        payload = {
            "domain": domain,
            "model": "your-model-name",
            "date": "YYYY-MM-DD",
            "responses": {case["id"]: "" for case in bench.get("cases", [])},
        }
        out = TEMPLATE_DIR / f"responses-{domain}.template.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out}")


def write_baseline(domain: str, example_path: Path) -> None:
    example = load_json(example_path)
    assistant = next(m["content"] for m in example["messages"] if m.get("role") == "assistant")
    bench = load_json(BENCH_DIR / f"benchmark-{domain}.json")
    responses = {case["id"]: assistant for case in bench.get("cases", [])}
    out = RESULTS_DIR / f"baseline-{domain}-seed.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"domain": domain, "responses": responses}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path = RESULTS_DIR / f"baseline-{domain}-seed.report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "score_benchmark.py"),
            str(out),
            "--domain",
            domain,
            "--out",
            str(report_path),
        ],
        check=False,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools/run_benchmark.py templates")
        print("  python tools/run_benchmark.py baseline [domain|all]")
        print("  python tools/run_benchmark.py score <responses.json> --domain philosophy")
        return 1

    command = sys.argv[1]
    if command == "templates":
        write_templates()
        return 0

    if command == "baseline":
        target = sys.argv[2] if len(sys.argv) > 2 else "philosophy"
        seeds = {
            "philosophy": ROOT / "training/examples/001-dao-de-jing-attribution.json",
            "psychology": ROOT / "training/examples/002-freud-cognitive-dissonance.json",
            "history": ROOT / "training/examples/003-marco-polo-pasta-myth.json",
            "religion": ROOT / "training/examples/004-confucian-ritual-council.json",
        }
        domains = DOMAINS if target == "all" else (target,)
        for domain in domains:
            path = seeds.get(domain)
            if path and path.exists():
                write_baseline(domain, path)
            else:
                print(f"Skip {domain}: missing seed {path}")
        return 0

    if command == "score":
        if len(sys.argv) < 3:
            print("Provide responses.json")
            return 1
        domain = "philosophy"
        if "--domain" in sys.argv:
            domain = sys.argv[sys.argv.index("--domain") + 1]
        out = RESULTS_DIR / f"latest-{domain}.report.json"
        return subprocess.call(
            [
                sys.executable,
                str(ROOT / "tools" / "score_benchmark.py"),
                sys.argv[2],
                "--domain",
                domain,
                "--out",
                str(out),
            ]
        )

    print(f"Unknown command: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())