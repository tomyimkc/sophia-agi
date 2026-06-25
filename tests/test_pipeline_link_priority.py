#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.link_priority (Phase 1).

Verifies registered-domain extraction (incl. multi-label suffixes), that a high-quality
site outranks a spammy one, that quotas scale with priority, and that the output is sorted
deterministically. Offline, no deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import document as docmod  # noqa: E402
from pipeline.link_priority import prioritize, registered_domain  # noqa: E402
from pipeline.quality_score import score_document  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "pipeline_docs.jsonl"


def test_registered_domain():
    assert registered_domain("https://a.b.example.com/x?y=1") == "example.com"
    assert registered_domain("https://plato.stanford.edu/entries/oikeiosis/") == "stanford.edu"
    assert registered_domain("https://news.bbc.co.uk/story") == "bbc.co.uk"
    assert registered_domain("https://example.com") == "example.com"
    assert registered_domain("not a url") == ""


def test_high_quality_site_outranks_spam():
    docs = list(docmod.read_jsonl(FIXTURES))
    for d in docs:
        d["quality"] = score_document(d)
    sites = prioritize(docs)
    rank = {s["domain"]: i for i, s in enumerate(sites)}
    # stanford.edu (consensus, two trusted sources) should outrank the spam blog.
    assert rank["stanford.edu"] < rank["randomblog.example"]
    # Sorted by descending priority.
    assert [s["priority"] for s in sites] == sorted((s["priority"] for s in sites), reverse=True)


def test_quota_scales_with_priority():
    docs = list(docmod.read_jsonl(FIXTURES))
    for d in docs:
        d["quality"] = score_document(d)
    sites = {s["domain"]: s for s in prioritize(docs)}
    assert sites["stanford.edu"]["suggestedQuota"] >= sites["randomblog.example"]["suggestedQuota"]
    assert all(s["suggestedQuota"] >= 1 for s in sites.values())


def test_aggregates_by_domain():
    docs = [
        {"url": "https://x.com/a", "content": "alpha " * 200, "quality": {"score": 0.8, "keep": True}},
        {"url": "https://x.com/b", "content": "beta " * 200, "quality": {"score": 0.6, "keep": True}},
    ]
    sites = prioritize(docs)
    assert len(sites) == 1
    assert sites[0]["domain"] == "x.com"
    assert sites[0]["docs"] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.link_priority tests passed")
