#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Compatibility wrapper for the learning-under-distribution-shift lane.

The implementation lives in tools/run_learning_shift.py. This wrapper exists so
operator docs can use the clearer command name:

    python tools/run_distribution_shift.py EXPERIMENT_SPEC.json --backend adapter

Use --template to emit a starter multi-case experiment spec scaffold.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = {
    "experimentId": "third-party-new-domain-YYYY-MM-DD",
    "oldBenchmarkBaselineScorePct": None,
    "learningRecords": [
        {
            "recordId": "new_domain_record_001",
            "domain": "new_domain",
            "text": "Verified source-grounded fact to learn.",
            "source": "source-pack://...",
            "confidence": "reviewed",
            "reviewerNote": "Human/domain reviewer approved.",
            "promoted": True,
        }
    ],
    "preTestPack": {
        "packId": "pre-new-domain",
        "visibility": "private-hidden",
        "cases": []
    },
    "postTestPack": {
        "packId": "post-new-domain-fresh",
        "visibility": "private-hidden",
        "cases": []
    },
    "oldBenchmarkPack": None,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", type=Path, nargs="?", help="experiment spec JSON for tools/run_learning_shift.py")
    ap.add_argument("--template", type=Path, help="write a starter spec template and exit")
    args, rest = ap.parse_known_args(argv)
    if args.template:
        args.template.parent.mkdir(parents=True, exist_ok=True)
        args.template.write_text(json.dumps(TEMPLATE, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.template}")
        return 0
    if not args.spec:
        ap.error("spec is required unless --template is provided")
    cmd = [sys.executable, str(ROOT / "tools" / "run_learning_shift.py"), str(args.spec), *rest]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
