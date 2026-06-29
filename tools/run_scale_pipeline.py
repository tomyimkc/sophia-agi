#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""End-to-end scale pipeline: WARC(s) -> dedup (KV) -> score -> sharded lake + catalog.

    python tools/run_scale_pipeline.py crawl.warc.gz --out-dir ./lake
    python tools/run_scale_pipeline.py warcs/ --out-dir ./lake --shard-size 5000 --keep-only

Runs the full data-engineering pipeline at scale against archived crawls, using the Phase 5
infra adapters (object store + persistent KV seen-set) so it behaves the same locally as on
real infra — point ``LocalObjectStore`` at a mount, or swap an S3/MinIO adapter, without
changing this script. Reports throughput, dedup ratio, and corpus stats — the numbers a data
team publishes. Offline, no API key.

Swap to cloud infra: replace ``LocalObjectStore(out_dir)`` with an S3/MinIO adapter
implementing ``pipeline.store.objectstore.ObjectStore``; the rest is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import corpus_table  # noqa: E402
from pipeline.dedup import minhash as mh  # noqa: E402
from pipeline.fetch.extract import extract_text  # noqa: E402
from pipeline.fetch.warc import read_warc, records_to_documents  # noqa: E402
from pipeline.quality_score import score_document  # noqa: E402
from pipeline.shard_writer import write_sharded  # noqa: E402
from pipeline.store.kv import SqliteSeenSet  # noqa: E402
from pipeline.store.objectstore import LocalObjectStore  # noqa: E402


def _iter_warcs(target: Path):
    if target.is_dir():
        yield from sorted(p for p in target.rglob("*") if p.suffix in (".warc", ".gz"))
    else:
        yield target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="a WARC file or a directory of WARC(.gz) files")
    ap.add_argument("--out-dir", required=True, help="object-store root (local filesystem lake)")
    ap.add_argument("--prefix", default="corpus")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--dedup-threshold", type=float, default=0.8)
    ap.add_argument("--keep-only", action="store_true", help="drop documents the scorer rejects")
    args = ap.parse_args(argv)

    store = LocalObjectStore(args.out_dir)
    seen = SqliteSeenSet(Path(args.out_dir) / "_seen.db")  # persistent URL+fingerprint dedup
    t0 = time.monotonic()

    raw_count = 0
    url_dups = 0
    bytes_in = 0
    kept_docs: list[dict] = []
    seen_minhash: dict[int, int] = {}

    for warc in _iter_warcs(Path(args.input)):
        for doc in records_to_documents(read_warc(warc)):
            raw_count += 1
            bytes_in += len(doc.get("content") or "")
            # Stage 1: exact URL dedup via the persistent KV seen-set.
            if not seen.add(doc["url"]):
                url_dups += 1
                continue
            doc["content"] = extract_text(doc["content"])
            kept_docs.append(doc)

    # Stage 2: near-dup content clustering (MinHash), keep one per cluster.
    cluster_ids, sigs = mh.cluster(
        [d["content"] for d in kept_docs], threshold=args.dedup_threshold
    )
    first_seen: set[int] = set()
    deduped: list[dict] = []
    near_dups = 0
    for i, doc in enumerate(kept_docs):
        cid = cluster_ids[i]
        doc["dedup"] = {"sim_cluster": f"c{cid}", "is_duplicate": cid in first_seen,
                        "minhash_sig": list(sigs[i])}
        if cid in first_seen:
            near_dups += 1
            continue
        first_seen.add(cid)
        deduped.append(doc)

    # Stage 3: score, optional quality filter.
    for doc in deduped:
        doc["quality"] = score_document(doc)
    if args.keep_only:
        deduped = [d for d in deduped if d["quality"]["keep"]]

    # Stage 4: write the sharded lake + catalog.
    catalog = write_sharded(
        deduped, store, prefix=args.prefix, shard_size=args.shard_size, pre_dedup_count=raw_count
    )
    summary = corpus_table.summarize(deduped)
    elapsed = max(1e-6, time.monotonic() - t0)

    print("=== scale pipeline report ===")
    print(f"input WARC docs      : {raw_count}")
    print(f"url duplicates       : {url_dups}")
    print(f"near-dup removed     : {near_dups}")
    print(f"final docs           : {summary['count']}")
    print(f"dedup ratio          : {round((raw_count - summary['count']) / raw_count, 4) if raw_count else 0}")
    print(f"tokens               : {summary['totalTokens']}")
    print(f"mean quality         : {summary['meanQuality']}  keepRate={summary['keepRate']}")
    print(f"shards               : {catalog['shardCount']} (size {args.shard_size})")
    print(f"bytes in             : {bytes_in}  ({bytes_in / 1e6:.2f} MB)")
    print(f"elapsed              : {elapsed:.3f}s   throughput {raw_count / elapsed:.1f} docs/s")
    print(f"lake                 : {args.out_dir}/{args.prefix}/_catalog.json")
    print(f"seen-set size        : {len(seen)}")
    seen.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
