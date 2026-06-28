# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Multimodal processing: perceptual-hash dedup + image-text quality (stretch).

``dedup_samples`` clusters near-duplicate images by perceptual-hash Hamming distance (greedy
single-linkage), the multimodal analogue of MinHash text dedup. ``score_sample`` scores an
image-text pair's training-worthiness from caption quality (length, boilerplate, alphanumeric
ratio) and provenance (reusing ``grounded_confidence`` priors), fail-closed on empty captions.

The phash for each sample comes from (in order) an explicit ``phash`` field, an
``image_matrix`` (synthetic/test images), or ``image_bytes`` decoded via PIL when available.
Samples without any image signal are left unclustered.
"""

from __future__ import annotations

import re

from agent.grounded_confidence import AUTHOR_CONFIDENCE_PRIOR, _DEFAULT_PRIOR
from pipeline.multimodal import phash as _ph

_WORD_RE = re.compile(r"[a-z0-9一-鿿]+", re.IGNORECASE)
_BOILERPLATE = ("click here", "buy now", "stock photo", "image may contain", "untitled")


def sample_phash(sample: dict) -> int | None:
    """Resolve a sample's perceptual hash, or None if no image signal is available."""
    if isinstance(sample.get("phash"), int):
        return sample["phash"]
    if sample.get("image_matrix"):
        return _ph.average_hash(sample["image_matrix"])
    if sample.get("image_bytes") and _ph.decode_available():
        return _ph.average_hash(_ph.image_to_matrix(sample["image_bytes"]))
    return None


def dedup_samples(samples, *, max_distance: int = 5) -> dict:
    """Cluster samples by perceptual-hash distance. Returns ``{kept, removed, stats}``.

    Samples within ``max_distance`` bits are duplicates; the first in each cluster is kept.
    Samples without a resolvable phash are kept (never silently dropped) and marked unhashed.
    """
    samples = list(samples)
    hashes = [sample_phash(s) for s in samples]
    reps: list[tuple[int, int]] = []  # (index, phash)
    kept, removed = [], []
    for i, s in enumerate(samples):
        h = hashes[i]
        if h is None:
            s["dedup"] = {"is_duplicate": False, "phash": None, "unhashed": True}
            kept.append(s)
            continue
        dup_of = None
        for rep_idx, rep_h in reps:
            if _ph.hamming(h, rep_h) <= max_distance:
                dup_of = rep_idx
                break
        if dup_of is None:
            reps.append((i, h))
            s["dedup"] = {"is_duplicate": False, "phash": h, "sim_cluster": f"c{i}"}
            kept.append(s)
        else:
            s["dedup"] = {"is_duplicate": True, "phash": h, "sim_cluster": f"c{dup_of}"}
            removed.append(s)
    n = len(samples)
    stats = {"input": n, "kept": len(kept), "removed": len(removed),
             "dedupRatio": round(len(removed) / n, 6) if n else 0.0}
    return {"kept": kept, "removed": removed, "stats": stats}


def _caption_quality(caption: str) -> tuple[float, list[str]]:
    words = _WORD_RE.findall(caption.lower())
    reasons: list[str] = []
    if len(words) < 3:
        return 0.1, ["caption too short (< 3 words)"]
    low = caption.lower()
    hits = sum(low.count(m) for m in _BOILERPLATE)
    non_space = [c for c in caption if not c.isspace()]
    alpha = sum(1 for c in non_space if c.isalnum()) / len(non_space) if non_space else 0.0
    length_score = min(1.0, len(words) / 12.0)
    boiler_score = max(0.0, 1.0 - hits / 2.0)
    if hits:
        reasons.append(f"{hits} boilerplate caption marker(s)")
    score = round(0.4 * length_score + 0.3 * boiler_score + 0.3 * alpha, 6)
    return score, reasons


def score_sample(sample: dict, *, keep_threshold: float = 0.5) -> dict:
    """Score an image-text pair. Returns ``{score, keep, signals, reasons}``."""
    caption = sample.get("caption") or ""
    cap_score, reasons = _caption_quality(caption)

    prov = sample.get("provenance") or {}
    tier = prov.get("authorConfidence")
    tier_prior = AUTHOR_CONFIDENCE_PRIOR.get(tier, _DEFAULT_PRIOR) if tier else None
    if tier_prior is not None:
        score = round(0.6 * cap_score + 0.4 * tier_prior, 6)
        reasons.append(f"provenance tier {tier!r} -> {tier_prior:.2f}")
    else:
        score = round(min(cap_score, 0.7), 6)  # no provenance -> capped (fail-closed)
        reasons.append("no provenance (capped)")

    return {
        "score": score,
        "keep": bool(score >= keep_threshold),
        "signals": {"caption": cap_score, "provenance": tier_prior},
        "reasons": reasons,
    }


__all__ = ["sample_phash", "dedup_samples", "score_sample"]
