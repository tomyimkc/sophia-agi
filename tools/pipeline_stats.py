#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Summarize a document shard and (optionally) gate it against a baseline.

    python tools/pipeline_stats.py tests/fixtures/pipeline_docs.jsonl
    python tools/pipeline_stats.py shard.parquet --out summary.json
    python tools/pipeline_stats.py shard.jsonl --baseline baseline.summary.json   # fail-closed gate

Reads a shard (JSONL, or Parquet when pyarrow is present), prints an analytical summary
(`pipeline.corpus_table`), and — when ``--baseline`` is given — runs the fail-closed
quality-regression gate (`pipeline.quality_regression`), exiting non-zero on any regression.
Offline, no API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import corpus_table, quality_regression  # noqa: E402


def _print_summary(s: dict) -> None:
    print(f"count={s['count']}  tokens={s['totalTokens']}  meanTokens={s['meanTokens']}")
    print(f"meanQuality={s['meanQuality']}  keepRate={s['keepRate']}  duplicateRate={s['duplicateRate']}")
    print(f"engine={s.get('engine', 'stdlib')}")
    print("langHistogram:", json.dumps(s["langHistogram"], ensure_ascii=False))
    print("qualityHistogram:", json.dumps(s["qualityHistogram"]))
    if s["domainCounts"]:
        print("topDomains:", json.dumps(dict(list(s["domainCounts"].items())[:5]), ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard", help="document shard (.jsonl or .parquet)")
    ap.add_argument("--out", help="write the summary JSON here")
    ap.add_argument("--baseline", help="baseline summary JSON to gate against (fail-closed)")
    args = ap.parse_args(argv)

    summary = corpus_table.summarize_shard(args.shard)
    _print_summary(summary)

    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nWrote summary -> {args.out}")

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        result = quality_regression.gate(baseline, summary)
        if result["ok"]:
            print("\nquality-regression gate: PASS")
            return 0
        print("\nquality-regression gate: FAIL (fail-closed)", file=sys.stderr)
        for p in result["problems"]:
            print(f"  - {p}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
