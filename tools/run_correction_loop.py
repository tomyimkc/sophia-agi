#!/usr/bin/env python3
"""Run correction loop on failed benchmark reports.

Usage:
  python tools/run_correction_loop.py --dry-run
  python tools/run_correction_loop.py --promote
  python tools/run_correction_loop.py --generate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.config import load_dotenv, normalize_api_keys  # noqa: E402
from agent.correction_loop import (  # noqa: E402
    draft_correction,
    find_failures,
    promote_corrections,
    write_pending,
)

RUNS = ROOT / "benchmark" / "model_runs"


def main() -> int:
    load_dotenv()
    normalize_api_keys()

    parser = argparse.ArgumentParser(description="Sophia correction loop")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--promote", action="store_true", help="Move pending corrections to training/examples")
    args = parser.parse_args()

    if args.promote:
        paths = promote_corrections()
        print(f"Promoted {len(paths)} correction(s)")
        return 0

    reports = sorted(RUNS.glob("*.report.json"))
    total_failures = 0
    for report_path in reports:
        failures = find_failures(report_path)
        if not failures:
            continue
        run_json = report_path.with_suffix("").with_suffix(".json")
        if run_json.name.endswith(".report.json"):
            run_json = Path(str(report_path).replace(".report.json", ".json"))
        responses = {}
        if run_json.exists():
            responses = json.loads(run_json.read_text(encoding="utf-8")).get("responses", {})

        print(f"{report_path.name}: {len(failures)} failure(s)")
        for failure in failures:
            total_failures += 1
            print(f"  - {failure['case_id']}: {failure['reasons']}")
            if args.generate and not args.dry_run:
                bad = responses.get(failure["case_id"], "")
                example = draft_correction(failure, bad)
                out = write_pending(example, failure["case_id"])
                print(f"    draft -> {out}")

    if total_failures == 0:
        print("No benchmark failures found.")
    elif args.dry_run:
        print(f"Dry run: {total_failures} failure(s) would get correction drafts with --generate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())