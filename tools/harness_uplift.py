#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measure harness uplift: passRate(model+harness) - passRate(model alone).

Holds the model fixed and grades both conditions with the same external verifier,
reporting the paired uplift with a bootstrap 95% CI. A positive point estimate is
NOT a demonstrated effect unless the CI lower bound clears 0 (no-overclaim gate).

    python tools/harness_uplift.py --provider mock
    python tools/harness_uplift.py suites/agent_smoke.json --provider deepseek --out eval/results/uplift.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.uplift import measure_uplift  # noqa: E402

DEFAULT_SUITE = [
    {"id": "advisor_decision", "goal": "Should we launch Sophia on Hacker News this week?", "mode": "advisor", "mustInclude": ["Decision"]},
    {"id": "repo_next_step", "goal": "What should we validate before the next release?", "mode": "repo", "mustInclude": ["Decision"]},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure harness uplift over a fixed model")
    parser.add_argument("suite", nargs="?", type=Path, default=None, help="suite JSON (defaults to a built-in smoke suite)")
    parser.add_argument("--provider", default="mock", help="model provider/preset (default mock for offline smoke)")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (reproducible CI)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    suite = json.loads(args.suite.read_text(encoding="utf-8")) if args.suite else DEFAULT_SUITE
    result = measure_uplift(suite, provider=args.provider, max_retries=args.max_retries, bootstrap_seed=args.seed)
    report = {"runAt": datetime.now().isoformat(timespec="seconds"), **result.to_dict()}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    verdict = "DEMONSTRATED" if result.demonstrated else "not demonstrated (CI includes 0)"
    print(f"\nUplift {result.uplift:+.3f}  CI95={result.uplift_ci95}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
