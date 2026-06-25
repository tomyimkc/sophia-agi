#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.corpus_table and pipeline.io (Phase 3).

Verifies the analytical summary over the fixtures (counts, token totals, language mix,
quality + domain histograms), JSONL shard round-trip, and — only when pyarrow/duckdb are
present — that the Parquet path returns the same headline numbers as the stdlib path.
Stdlib core; heavy engines optional. Offline, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import corpus_table, document as docmod, io as pio  # noqa: E402
from pipeline.quality_score import score_document  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "pipeline_docs.jsonl"


def _scored():
    docs = list(docmod.read_jsonl(FIXTURES))
    for d in docs:
        d["quality"] = score_document(d)
    return docs


def test_summary_shape_and_counts():
    s = corpus_table.summarize(_scored())
    assert s["count"] == 6
    assert s["totalTokens"] > 0
    assert s["meanQuality"] is not None
    assert 0.0 <= s["keepRate"] <= 1.0
    # Languages: 5 en + 1 zh in the fixtures.
    assert s["langHistogram"].get("zh") == 1
    assert s["langHistogram"].get("en") == 5
    # Quality histogram buckets sum to the doc count.
    assert sum(s["qualityHistogram"].values()) == 6


def test_domain_counts():
    s = corpus_table.summarize(_scored())
    assert s["domainCounts"].get("wikipedia.org") == 2
    assert s["domainCounts"].get("stanford.edu") == 1


def test_empty_corpus():
    s = corpus_table.summarize([])
    assert s["count"] == 0
    assert s["meanQuality"] is None
    assert s["keepRate"] == 0.0


def test_jsonl_shard_roundtrip():
    import tempfile

    docs = _scored()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "shard.jsonl"
        n = pio.write_shard(p, docs)
        assert n == 6
        back = pio.read_shard(p)
        assert len(back) == 6
        assert corpus_table.summarize_shard(p)["count"] == 6


def test_parquet_path_matches_stdlib_if_available():
    if not pio.parquet_available():
        print("  (pyarrow unavailable — Parquet path skipped)")
        return
    import tempfile

    docs = _scored()
    stdlib_summary = corpus_table.summarize(docs)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "shard.parquet"
        pio.write_shard(p, docs)
        pq_summary = corpus_table.summarize_shard(p)
    # Headline numbers must match regardless of engine.
    for key in ("count", "totalTokens", "keepRate", "duplicateRate"):
        assert pq_summary[key] == stdlib_summary[key]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.corpus_table tests passed")
