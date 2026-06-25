# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shard manifest / catalog for the pipeline (Phase 0).

A manifest is the self-describing catalog entry for a batch of documents after they pass
through the pipeline: how many rows, what fraction survived dedup, the distribution of
quality scores, and a content hash for reproducibility. It mirrors the pattern already used
by ``rag/index/embeddings.meta.json`` (count + sha256 so a rebuild can be verified).

Deterministic and stdlib-only: the same set of documents always yields the same manifest
(``sort_keys`` + sorted hashing), so a manifest can be committed and checked in CI.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from pipeline import SCHEMA_VERSION

#: Histogram bucket edges for quality scores in [0,1] -> 5 buckets.
_QUALITY_BUCKETS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001))


def _quality_histogram(docs: Sequence[dict]) -> dict[str, int]:
    """Bucket each doc's ``quality.score`` into 5 fixed bins (missing score -> 'unscored')."""
    hist = {f"{lo:.1f}-{min(hi, 1.0):.1f}": 0 for lo, hi in _QUALITY_BUCKETS}
    hist["unscored"] = 0
    for doc in docs:
        score = (doc.get("quality") or {}).get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            hist["unscored"] += 1
            continue
        for lo, hi in _QUALITY_BUCKETS:
            if lo <= float(score) < hi:
                hist[f"{lo:.1f}-{min(hi, 1.0):.1f}"] += 1
                break
    return hist


def content_sha256(docs: Sequence[dict]) -> str:
    """A deterministic content hash over the documents (order-independent).

    Each doc is canonicalized with ``sort_keys`` and the per-doc digests are sorted before
    hashing, so reordering the shard does not change the manifest hash.
    """
    digests = sorted(
        hashlib.sha256(json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        for d in docs
    )
    h = hashlib.sha256()
    for d in digests:
        h.update(d.encode("ascii"))
    return h.hexdigest()


def build_manifest(
    docs: Sequence[dict],
    *,
    shard_path: str | None = None,
    pre_dedup_count: int | None = None,
) -> dict:
    """Build a manifest for a shard of documents.

    ``pre_dedup_count`` (rows before the dedup stage dropped duplicates) lets the manifest
    record a ``dedup_ratio``; pass ``len(docs)`` or omit if no dedup ran.
    """
    kept = list(docs)
    n = len(kept)
    duplicates_removed = None
    dedup_ratio = None
    if pre_dedup_count is not None and pre_dedup_count > 0:
        duplicates_removed = max(0, pre_dedup_count - n)
        dedup_ratio = round(duplicates_removed / pre_dedup_count, 6)

    manifest = {
        "schema": SCHEMA_VERSION,
        "shardPath": shard_path,
        "rowCount": n,
        "preDedupCount": pre_dedup_count,
        "duplicatesRemoved": duplicates_removed,
        "dedupRatio": dedup_ratio,
        "qualityHistogram": _quality_histogram(kept),
        "contentSha256": content_sha256(kept),
    }
    return manifest


def write_manifest(path: str | Path, manifest: dict) -> Path:
    """Write a manifest to ``path`` as pretty, deterministic JSON."""
    p = Path(path)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def read_manifest(path: str | Path) -> dict:
    """Load a manifest JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(docs: Sequence[dict], manifest: dict) -> list[str]:
    """Fail-closed check that ``docs`` still match a committed ``manifest``.

    Returns a list of mismatches (empty == verified). Used by CI to assert a shard hasn't
    drifted from its catalog entry (same contract as ``build_rag_index.py --verify``).
    """
    problems: list[str] = []
    rebuilt = build_manifest(docs, shard_path=manifest.get("shardPath"), pre_dedup_count=manifest.get("preDedupCount"))
    if rebuilt["rowCount"] != manifest.get("rowCount"):
        problems.append(f"rowCount {rebuilt['rowCount']} != manifest {manifest.get('rowCount')}")
    if rebuilt["contentSha256"] != manifest.get("contentSha256"):
        problems.append("contentSha256 mismatch (shard content drifted from manifest)")
    return problems


__all__ = [
    "build_manifest",
    "write_manifest",
    "read_manifest",
    "verify_manifest",
    "content_sha256",
]
