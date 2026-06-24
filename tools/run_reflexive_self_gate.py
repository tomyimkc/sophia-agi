#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run Sophia's reflexive no-overclaim AGI self-gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reflexive_self_gate import scan_paths  # noqa: E402

DEFAULT_PATHS = ["README.md", "RESULTS.md", "agi-proof", "docs/06-Roadmap", "docs/11-Platform"]
DEFAULT_OUT = ROOT / "agi-proof" / "self-gate" / "reflexive-self-gate.public-report.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", default=DEFAULT_PATHS)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    report = scan_paths(args.paths, repo_root=ROOT)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "verdict": report["verdict"], "summary": report["summary"], "canClaimAGI": report["canClaimAGI"]}, indent=2))
    return 0 if report["verdict"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
