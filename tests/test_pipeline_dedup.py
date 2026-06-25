#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.dedup (Phase 2): MinHash-LSH near-dup + the dedup stage.

Verifies that identical and near-identical texts cluster together while distinct texts do
not, that Jaccard estimates are sane, that signatures are deterministic, that the mirror
pair in the fixtures is detected as a duplicate, and that the dedup stage sets the dedup
block and computes a dedup ratio. Vector dedup is exercised only if numpy is present.
Offline, stdlib-only core.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import document as docmod  # noqa: E402
from pipeline.dedup import dedup_documents  # noqa: E402
from pipeline.dedup import minhash as mh  # noqa: E402
from pipeline.dedup import vector as vec  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "pipeline_docs.jsonl"

_BASE = (
    "Stoicism is a school of Hellenistic philosophy founded by Zeno of Citium in Athens in "
    "the early third century BC. It teaches that virtue is the highest good and that the wise "
    "live in agreement with nature, practicing the cardinal virtues to reach eudaimonia."
)
_NEAR = _BASE.replace("the highest good", "the supreme good")  # one-word edit
_DIFFERENT = (
    "Quantum chromodynamics describes the strong interaction between quarks and gluons via "
    "color charge, an essential pillar of the Standard Model of particle physics."
)


def test_identical_cluster_together():
    # Identical copies always cluster; unrelated text stays separate.
    ids, _sigs = mh.cluster([_BASE, _BASE, _DIFFERENT], threshold=0.8)
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


def test_minhash_estimate_tracks_true_jaccard():
    # A one-word edit in a ~45-word doc is ~0.78 Jaccard; MinHash estimates it closely,
    # so it clusters at threshold 0.7 but (correctly) not at the stricter 0.8.
    est = mh.jaccard_estimate(mh.signature(_BASE), mh.signature(_NEAR))
    assert 0.7 <= est <= 0.85
    loose_ids = mh.cluster([_BASE, _NEAR], threshold=0.7)[0]
    assert loose_ids[0] == loose_ids[1]
    strict_ids = mh.cluster([_BASE, _NEAR], threshold=0.8)[0]
    assert strict_ids[0] != strict_ids[1]


def test_jaccard_estimate_bounds():
    sig = mh.signature(_BASE)
    assert mh.jaccard_estimate(sig, sig) == 1.0
    assert mh.jaccard_estimate(mh.signature(_BASE), mh.signature(_DIFFERENT)) < 0.3


def test_signature_deterministic():
    assert mh.signature(_BASE) == mh.signature(_BASE)


def test_mirror_pair_detected_in_fixtures():
    docs = list(docmod.read_jsonl(FIXTURES))
    result = dedup_documents(docs)
    # The wikipedia mirror (identical content, different host + tracking params) is a dup.
    urls_removed = {d["url"] for d in result["removed"]}
    assert any("mirror" in u for u in urls_removed)
    assert result["stats"]["removed"] >= 1
    assert 0.0 < result["stats"]["dedupRatio"] <= 1.0
    # Every doc got a dedup block and a canonical_url.
    for d in docs:
        assert "is_duplicate" in d["dedup"]
        assert d["canonical_url"]


def test_dedup_stage_keeps_one_per_cluster():
    docs = [
        {"url": "https://a.com/1", "content": _BASE},
        {"url": "https://b.com/2", "content": _BASE},  # mirror
        {"url": "https://c.com/3", "content": _DIFFERENT},
    ]
    result = dedup_documents(docs)
    assert result["stats"]["kept"] == 2
    assert result["stats"]["removed"] == 1


def test_vector_dedup_if_numpy_present():
    if not vec.available():
        print("  (numpy/embedder unavailable — vector dedup skipped)")
        return
    ids = vec.cluster_documents(
        [{"content": _BASE}, {"content": _NEAR}, {"content": _DIFFERENT}], threshold=0.8
    )
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.dedup tests passed")
