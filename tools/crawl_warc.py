#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ingest a WARC archive through the full pipeline: extract → dedup → score → shard.

    python tools/crawl_warc.py crawl.warc            # or crawl.warc.gz
    python tools/crawl_warc.py crawl.warc --out shard.jsonl --manifest shard.manifest.json

Reads ``response`` records from a WARC (CommonCrawl-style), strips HTML to text, removes
near-duplicates (URL canonicalization + MinHash), scores each document for training-worthiness
(provenance + heuristics), and reports a corpus summary. This is the network-free way to run
the acquisition pipeline at scale against an archived crawl. Offline, no API key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import corpus_table, manifest as man  # noqa: E402
from pipeline.dedup import dedup_documents  # noqa: E402
from pipeline.fetch.extract import extract_text  # noqa: E402
from pipeline.fetch.warc import read_warc, records_to_documents  # noqa: E402
from pipeline.quality_score import score_document  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("warc", help="WARC file (.warc or .warc.gz)")
    ap.add_argument("--out", help="write the scored, deduped shard here (JSONL)")
    ap.add_argument("--manifest", help="write a shard manifest here")
    ap.add_argument("--dedup-threshold", type=float, default=0.8)
    ap.add_argument("--keep-only", action="store_true", help="drop documents the scorer rejects")
    args = ap.parse_args(argv)

    raw_docs = []
    for doc in records_to_documents(read_warc(args.warc)):
        doc["content"] = extract_text(doc["content"])
        raw_docs.append(doc)
    pre = len(raw_docs)
    print(f"WARC response docs: {pre}")

    result = dedup_documents(raw_docs, threshold=args.dedup_threshold)
    docs = result["kept"]
    print(
        f"Dedup: kept {result['stats']['kept']}/{pre} "
        f"(removed {result['stats']['removed']}, ratio {result['stats']['dedupRatio']:.3f})"
    )

    for doc in docs:
        doc["quality"] = score_document(doc)
    if args.keep_only:
        docs = [d for d in docs if d["quality"]["keep"]]
        print(f"Quality filter: {len(docs)} kept")

    summary = corpus_table.summarize(docs)
    print(
        f"Summary: count={summary['count']} tokens={summary['totalTokens']} "
        f"meanQuality={summary['meanQuality']} keepRate={summary['keepRate']}"
    )

    if args.out:
        from pipeline import document as docmod

        n = docmod.write_jsonl(args.out, docs)
        print(f"Wrote {n} docs -> {args.out}")
    if args.manifest:
        m = man.build_manifest(docs, shard_path=args.out, pre_dedup_count=pre)
        man.write_manifest(args.manifest, m)
        print(f"Wrote manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
