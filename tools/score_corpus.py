#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Score a JSONL document batch and report quality + per-site crawl priority.

    python tools/score_corpus.py tests/fixtures/pipeline_docs.jsonl
    python tools/score_corpus.py in.jsonl --out scored.jsonl --manifest shard.manifest.json

Reads documents (``pipeline.document`` contract), assigns each a quality block via
``pipeline.quality_score.score_document`` (provenance + heuristics, offline & deterministic),
prints a quality histogram and a priority-ranked site table (``pipeline.link_priority``), and
optionally writes the scored corpus and a shard manifest. No network, no API key.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import document as docmod  # noqa: E402
from pipeline import link_priority, manifest, quality_score  # noqa: E402


def _print_histogram(docs: list[dict]) -> None:
    m = manifest.build_manifest(docs)
    print(f"\nDocuments: {m['rowCount']}")
    print("Quality histogram:")
    for bucket, count in sorted(m["qualityHistogram"].items()):
        bar = "#" * count
        print(f"  {bucket:>10}: {count:>4} {bar}")


def _print_sites(docs: list[dict], base_quota: int) -> None:
    sites = link_priority.prioritize(docs, base_quota=base_quota)
    print("\nSite priority (descending):")
    print(f"  {'domain':<32} {'docs':>4} {'kept':>4} {'meanQ':>7} {'prio':>6} {'quota':>6}")
    for s in sites:
        print(
            f"  {s['domain']:<32} {s['docs']:>4} {s['kept']:>4} "
            f"{s['meanQuality']:>7.3f} {s['priority']:>6.3f} {s['suggestedQuota']:>6}"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input JSONL of documents")
    ap.add_argument("--out", help="write scored JSONL here")
    ap.add_argument("--manifest", help="write a shard manifest here")
    ap.add_argument("--keep-threshold", type=float, default=quality_score.DEFAULT_KEEP_THRESHOLD)
    ap.add_argument("-k", type=int, default=2, help="independent-corroboration floor (poison gate)")
    ap.add_argument("--base-quota", type=int, default=100)
    ap.add_argument("--quiet", action="store_true", help="suppress per-document reasons")
    args = ap.parse_args(argv)

    raw = list(docmod.read_jsonl(args.input))
    bad = 0
    docs: list[dict] = []
    for i, doc in enumerate(raw):
        problems = docmod.validate(doc)
        if problems:
            bad += 1
            print(f"[skip] row {i}: {problems[0]}", file=sys.stderr)
            continue
        doc["quality"] = quality_score.score_document(
            doc, k=args.k, keep_threshold=args.keep_threshold
        )
        docs.append(doc)

    if not args.quiet:
        print("Per-document scores:")
        for doc in docs:
            q = doc["quality"]
            flag = "KEEP" if q["keep"] else "drop"
            print(f"  [{flag}] {q['score']:.3f}  {doc['url']}")

    _print_histogram(docs)
    _print_sites(docs, args.base_quota)
    if bad:
        print(f"\n{bad} row(s) skipped (failed document contract).", file=sys.stderr)

    if args.out:
        n = docmod.write_jsonl(args.out, docs)
        print(f"\nWrote {n} scored docs -> {args.out}")
    if args.manifest:
        m = manifest.build_manifest(docs, shard_path=args.out, pre_dedup_count=len(raw))
        manifest.write_manifest(args.manifest, m)
        print(f"Wrote manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
