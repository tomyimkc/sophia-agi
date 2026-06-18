#!/usr/bin/env python3
"""Run the Sophia Attribution Benchmark workflow."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "tests" / "attribution_bench.json"
TEMPLATE_PATH = ROOT / "benchmark" / "responses.template.json"
RESULTS_DIR = ROOT / "benchmark" / "results"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_template() -> None:
    bench = load_json(BENCH_PATH)
    payload = {
        "model": "your-model-name",
        "date": "YYYY-MM-DD",
        "responses": {case["id"]: "" for case in bench.get("cases", [])},
    }
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote template: {TEMPLATE_PATH}")


def write_baseline() -> None:
    """Score the seed training example as a naive baseline reference."""
    example_path = ROOT / "training" / "examples" / "001-dao-de-jing-attribution.json"
    example = load_json(example_path)
    assistant = next(
        m["content"] for m in example["messages"] if m.get("role") == "assistant"
    )
    bench = load_json(BENCH_PATH)
    responses = {case["id"]: assistant for case in bench.get("cases", [])}
    out = RESULTS_DIR / "baseline-seed-example.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"responses": responses}, indent=2) + "\n", encoding="utf-8")

    score_script = ROOT / "tools" / "score_benchmark.py"
    report_path = RESULTS_DIR / "baseline-seed-example.report.json"
    subprocess.run(
        [sys.executable, str(score_script), str(out), "--out", str(report_path)],
        check=False,
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tools/run_benchmark.py template   # create response template")
        print("  python tools/run_benchmark.py baseline   # score seed example")
        print("  python tools/run_benchmark.py score <responses.json>")
        return 1

    command = sys.argv[1]
    if command == "template":
        write_template()
        return 0
    if command == "baseline":
        write_baseline()
        return 0
    if command == "score":
        if len(sys.argv) < 3:
            print("Provide responses.json path")
            return 1
        score_script = ROOT / "tools" / "score_benchmark.py"
        out = RESULTS_DIR / "latest.report.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        return subprocess.call(
            [sys.executable, str(score_script), sys.argv[2], "--out", str(out)]
        )

    print(f"Unknown command: {command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())