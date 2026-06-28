#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.quality_score (Phase 1).

Verifies that a well-sourced, well-formed document outscores a single-low-trust-source
spammy one; that boilerplate/short pages score low; that a no-provenance document is capped
below the top tier (fail-closed); that a poison-gate quarantine caps the score and forces
drop; that scoring is deterministic; and that CJK content is handled. Offline, no deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import document as docmod  # noqa: E402
from pipeline.quality_score import score_document  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "pipeline_docs.jsonl"


def _load():
    return {d["url"]: d for d in docmod.read_jsonl(FIXTURES)}


def test_high_quality_beats_low_trust_spam():
    docs = _load()
    good = score_document(docs["https://plato.stanford.edu/entries/oikeiosis/"])
    spam = score_document(docs["https://randomblog.example/post?utm_source=fb&utm_campaign=x"])
    assert good["score"] > spam["score"]
    assert good["keep"] is True
    assert spam["keep"] is False


def test_single_low_trust_source_is_quarantined_and_capped():
    docs = _load()
    spam = score_document(docs["https://randomblog.example/post?utm_source=fb&utm_campaign=x"])
    # One source, trust 0.15 -> cannot meet k>=2; poison gate quarantines, score capped, drop.
    assert any("quarantine" in r for r in spam["reasons"])
    assert spam["keep"] is False
    assert spam["score"] <= 0.7


def test_boilerplate_scores_low():
    docs = _load()
    ads = score_document(docs["https://ads.example/lp"])
    assert ads["signals"]["boilerplate"] < 0.6
    assert ads["score"] < 0.6
    assert ads["keep"] is False


def test_no_provenance_is_capped():
    doc = {
        "url": "https://example.org/clean",
        "content": (
            "This is a long, clean, well-formed paragraph of ordinary prose with no spam "
            "markers and plenty of distinct vocabulary describing a perfectly reasonable "
            "topic in enough detail to clear the length threshold comfortably and then some."
        ),
    }
    q = score_document(doc)
    # Good content but no sources at all -> cannot exceed the no-provenance cap.
    assert q["score"] <= 0.7
    assert any("capped" in r for r in q["reasons"])


def test_cjk_document_handled():
    docs = _load()
    zh = score_document(docs["https://zh.wikipedia.org/wiki/斯多葛主义"])
    assert zh["signals"]["script_purity"] > 0.9
    assert zh["keep"] is True


def test_deterministic():
    docs = _load()
    d = docs["https://en.wikipedia.org/wiki/Stoicism"]
    assert score_document(d) == score_document(d)


def test_quality_block_validates():
    docs = _load()
    for d in docs.values():
        d["quality"] = score_document(d)
        assert docmod.validate(d) == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.quality_score tests passed")
