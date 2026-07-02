#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""ANN-crossover tripwire for the memory stack.

Brute-force cosine over the committed npz index is the *correct* choice below the
approximate-nearest-neighbor crossover (external vendor-reported benchmarks put it
around ~5k records; below that ANN ties or loses while adding nondeterminism and a
dependency). This test is the tripwire that makes the upgrade trigger itself: when
the corpus outgrows the threshold, the failure message says exactly what to do —
adopt a deterministic-build ANN backend (e.g. hnswlib with fixed seed/M/ef and the
committed-manifest verify pattern of ``tools/build_rag_index.py``) instead of
silently eating linear scans.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "rag" / "index" / "embeddings.meta.json"
CHUNKS = ROOT / "rag" / "index" / "chunks.jsonl"

ANN_CROSSOVER = 5000


def test_rag_index_below_ann_crossover() -> None:
    if not META.exists():
        return  # no committed index in this checkout — nothing to trip on
    count = int(json.loads(META.read_text(encoding="utf-8")).get("count", 0))
    assert count < ANN_CROSSOVER, (
        f"RAG index has {count} chunks >= ANN crossover ({ANN_CROSSOVER}): brute-force "
        f"cosine is no longer the right default. Adopt an ANN backend behind "
        f"agent/vector_store.py with a DETERMINISTIC build (fixed seed/M/ef) and a "
        f"committed manifest hash, mirroring tools/build_rag_index.py --verify. "
        f"See docs/06-Roadmap/Ruflo-Integration-Research-2026-07-02.md §3.3."
    )


def test_chunks_and_meta_agree() -> None:
    """The tripwire only works if `count` is honest — cross-check it against the
    committed chunks file when both exist."""
    if not (META.exists() and CHUNKS.exists()):
        return
    count = int(json.loads(META.read_text(encoding="utf-8")).get("count", 0))
    n_lines = sum(1 for line in CHUNKS.read_text(encoding="utf-8").splitlines()
                  if line.strip())
    assert count == n_lines, (
        f"embeddings.meta.json count={count} but chunks.jsonl has {n_lines} rows — "
        f"regenerate the index (tools/build_rag_index.py)"
    )
