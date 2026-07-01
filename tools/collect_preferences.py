#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Distill harness run traces into preference pairs (the data half of the
model<->harness co-evolution loop).

Reads append-only run logs (default: the harness RUNS_DIR) and emits JSONL
preference records — each a fail-then-fixed step, with the passing attempt as
`chosen` and an earlier failing attempt as `rejected`. Produces the dataset only;
training is a separate, separately-gated job.

    python tools/collect_preferences.py                      # scan agent/memory/agent_runs
    python tools/collect_preferences.py --runs-dir path/ --out training/preferences.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import harness  # noqa: E402
from agent.trace_distill import distill_dir, to_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill harness traces into preference pairs")
    parser.add_argument("--runs-dir", type=Path, default=harness.RUNS_DIR, help="directory of *.jsonl run logs")
    parser.add_argument("--out", type=Path, default=None, help="write JSONL here (else stdout)")
    args = parser.parse_args()

    pairs = distill_dir(args.runs_dir)
    payload = to_jsonl(pairs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        print(f"Wrote {len(pairs)} preference pair(s) to {args.out}")
    else:
        if payload:
            print(payload)
        print(f"# {len(pairs)} preference pair(s) from {args.runs_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
