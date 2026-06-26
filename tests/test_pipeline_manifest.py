#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.manifest and pipeline.document (Phase 0).

Verifies the document contract validator (required fields, ranges), source extraction, the
deterministic order-independent content hash, dedup-ratio accounting, manifest round-trip,
and fail-closed verification against content drift. Offline, no deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import document as docmod  # noqa: E402
from pipeline import manifest as man  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "pipeline_docs.jsonl"


def test_fixtures_validate():
    docs = list(docmod.read_jsonl(FIXTURES))
    assert len(docs) == 6
    for d in docs:
        assert docmod.validate(d) == []


def test_validate_catches_bad_docs():
    assert docmod.validate({}) == ["missing or empty 'url'", "missing or non-string 'content'"]
    assert "unknown provenance.authorConfidence tier" in " ".join(
        docmod.validate({"url": "u", "content": "c", "provenance": {"authorConfidence": "bogus"}})
    )
    assert "must be a probability" in " ".join(
        docmod.validate(
            {"url": "u", "content": "c", "provenance": {"sources": [{"sourceId": "s", "trust": 9}]}}
        )
    )


def test_to_sources():
    doc = {"url": "u", "content": "c", "provenance": {"sources": [{"confidence": 0.8}]}}
    srcs = docmod.to_sources(doc, default_trust=0.4)
    assert srcs[0]["sourceId"] == "_src0"
    assert srcs[0]["trust"] == 0.4
    assert docmod.to_sources({"url": "u", "content": "c"}) == []


def test_content_hash_order_independent():
    docs = list(docmod.read_jsonl(FIXTURES))
    assert man.content_sha256(docs) == man.content_sha256(list(reversed(docs)))


def test_dedup_ratio():
    docs = list(docmod.read_jsonl(FIXTURES))[:4]
    m = man.build_manifest(docs, pre_dedup_count=6)
    assert m["rowCount"] == 4
    assert m["duplicatesRemoved"] == 2
    assert m["dedupRatio"] == round(2 / 6, 6)


def test_manifest_roundtrip_and_verify():
    import tempfile

    docs = list(docmod.read_jsonl(FIXTURES))
    m = man.build_manifest(docs, shard_path="shard.jsonl")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "shard.manifest.json"
        man.write_manifest(p, m)
        loaded = man.read_manifest(p)
        assert loaded["contentSha256"] == m["contentSha256"]
        assert man.verify_manifest(docs, loaded) == []
        # Drift: drop a doc -> verification fails.
        problems = man.verify_manifest(docs[:-1], loaded)
        assert problems and "mismatch" in " ".join(problems).lower()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.manifest tests passed")
