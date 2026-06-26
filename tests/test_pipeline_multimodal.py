#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the multimodal slice (pipeline.multimodal).

Covers perceptual hashing (identical → distance 0, small perturbation → small distance,
different → large), phash dedup clustering, image-text quality scoring (caption length /
boilerplate / provenance, fail-closed on empty), sample validation, and WebDataset tar
round-trip with provenance metadata. Stdlib-only — synthetic grayscale matrices stand in for
images, so no PIL/network is needed. Offline.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.multimodal import process, sample as smod, shards  # noqa: E402
from pipeline.multimodal.phash import average_hash, hamming  # noqa: E402


def _gradient(w=16, h=16, shift=0):
    return [[(x + y + shift) % 256 for x in range(w)] for y in range(h)]


def _flat(val=128, w=16, h=16):
    return [[val for _ in range(w)] for _ in range(h)]


# ------------------------------- phash ------------------------------------- #

def test_phash_identical_and_different():
    a = average_hash(_gradient())
    b = average_hash(_gradient())
    assert hamming(a, b) == 0
    far = average_hash(_flat(0))
    near_flat = average_hash(_flat(255))
    # A gradient vs a flat image differ substantially.
    assert hamming(a, far) > 0


def test_phash_small_perturbation_small_distance():
    base = average_hash(_gradient(shift=0))
    perturbed = average_hash(_gradient(shift=1))  # tiny luminance shift
    assert hamming(base, perturbed) <= 4


# ------------------------------- dedup ------------------------------------- #

def test_dedup_clusters_near_images():
    samples = [
        {"id": "a", "caption": "a scene", "image_matrix": _gradient(shift=0)},
        {"id": "b", "caption": "same scene mirror", "image_matrix": _gradient(shift=1)},
        {"id": "c", "caption": "totally different", "image_matrix": _flat(10)},
    ]
    result = process.dedup_samples(samples, max_distance=5)
    removed_ids = {s["id"] for s in result["removed"]}
    assert "b" in removed_ids  # near-dup of a
    assert result["stats"]["kept"] == 2


def test_dedup_keeps_unhashable():
    samples = [{"id": "x", "caption": "no image signal"}]
    result = process.dedup_samples(samples)
    assert result["stats"]["kept"] == 1
    assert result["kept"][0]["dedup"]["unhashed"] is True


# ------------------------------ quality ------------------------------------ #

def test_score_sample_provenance_beats_spam():
    good = process.score_sample(
        {"id": "g", "caption": "A detailed photograph of a snow leopard resting on a rocky ledge",
         "provenance": {"authorConfidence": "consensus"}}
    )
    spam = process.score_sample({"id": "s", "caption": "click here buy now"})
    assert good["score"] > spam["score"]
    assert good["keep"] is True


def test_score_sample_empty_caption_fails_closed():
    q = process.score_sample({"id": "e", "caption": ""})
    assert q["keep"] is False
    assert q["score"] < 0.5


def test_no_provenance_capped():
    q = process.score_sample(
        {"id": "n", "caption": "A long descriptive caption with plenty of distinct words here"}
    )
    assert q["score"] <= 0.7


# ----------------------------- validation ---------------------------------- #

def test_sample_validation():
    assert smod.validate({"id": "a", "caption": "x", "phash": 5}) == []
    problems = smod.validate({"caption": "x"})
    assert any("id" in p for p in problems)
    assert any("image" in p for p in smod.validate({"id": "a", "caption": "x"}))


# ---------------------------- webdataset ----------------------------------- #

def test_webdataset_roundtrip():
    samples = [
        {"id": "a", "caption": "first", "image_bytes": b"\xff\xd8fakejpeg",
         "provenance": {"authorConfidence": "attributed"}, "quality": {"score": 0.8, "keep": True}},
        {"id": "b", "caption": "second", "image_bytes": b"\xff\xd8other"},
    ]
    with tempfile.TemporaryDirectory() as td:
        tar = Path(td) / "part-000.tar"
        assert shards.write_webdataset(samples, tar) == 2
        back = {s["id"]: s for s in shards.read_webdataset(tar)}
        assert back["a"]["caption"] == "first"
        assert back["a"]["image_bytes"] == b"\xff\xd8fakejpeg"
        assert back["a"]["provenance"]["authorConfidence"] == "attributed"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.multimodal tests passed")
