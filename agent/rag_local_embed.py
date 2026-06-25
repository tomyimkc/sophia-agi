# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Local, deterministic, dependency-light embedding backend for Sophia RAG.

The committed RAG scaffold (`agent/vector_store.py`) already does cosine search when an
`embeddings.npz` is present, but the only embedder shipped (`agent/rag_embed.py`) calls
the Gemini/Vertex API — so offline/airgapped Sophia had **no** way to populate the index,
and vector recall stayed dead scaffolding (no `embeddings.npz` was committed).

This module closes that with a **hashing embedding**: it maps char n-grams + word tokens
into a fixed-width vector by hashing each feature to a bucket (signed, `blake2b` — not the
salted builtin ``hash``, so it is stable across runs, processes, and platforms), applies
sublinear term weighting, and L2-normalizes. Cosine similarity over these vectors is a real
vector-space recall that catches morphological / substring overlap the exact-token keyword
scorer misses, while being:

  - **offline & CPU-only** — numpy is the only dependency (already required by RAG);
  - **deterministic & reproducible** — same input → byte-identical vectors → stable hash;
  - **provider-free** — no API key, works under ``SOPHIA_PROFILE=airgap``.

Honest bound: this is a *lexical-semantic hash* embedding, **not** a learned neural one.
It generalizes over surface form, not deep meaning; for true semantic recall the Gemini
backend (`agent/rag_embed.py`) remains the higher-quality option when a key is available.
The index records which backend produced it (see `tools/build_rag_index.py`) so queries are
always embedded with the matching embedder.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

#: Identifier stamped into the index so retrieval embeds queries with the SAME backend.
BACKEND_ID = "local-hash-v1"
#: Vector width. Larger = fewer hash collisions; 1024 is ample for a few-thousand-chunk corpus.
DIM = 1024

_WORD_RE = re.compile(r"[a-z0-9一-鿿]+")


def _features(text: str) -> "list[str]":
    """Word unigrams + character trigrams (incl. CJK), lowercased.

    Char trigrams give graceful overlap on morphology/typos/substrings; CJK is split
    per-character (no whitespace) so trigrams still capture local context.
    """
    low = text.lower()
    words = _WORD_RE.findall(low)
    feats: list[str] = [f"w:{w}" for w in words]
    for w in words:
        padded = f"#{w}#"
        for i in range(len(padded) - 2):
            feats.append(f"t:{padded[i:i + 3]}")
    return feats


def _bucket_and_sign(feature: str) -> "tuple[int, float]":
    h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(h, "big")
    bucket = val % DIM
    sign = 1.0 if (val >> 63) & 1 else -1.0
    return bucket, sign


def embed_text(text: str) -> np.ndarray:
    """Embed one string into an L2-normalized ``float32`` vector of width :data:`DIM`."""
    counts: dict[int, float] = {}
    signs: dict[int, float] = {}
    raw: dict[str, int] = {}
    for feat in _features(text or ""):
        raw[feat] = raw.get(feat, 0) + 1
    for feat, c in raw.items():
        bucket, sign = _bucket_and_sign(feat)
        # sublinear term weighting (1 + log tf) dampens repeated-token dominance
        counts[bucket] = counts.get(bucket, 0.0) + (1.0 + math.log(c)) * sign
        signs[bucket] = sign
    # Normalize with a pure-Python L2 norm (math.fsum, not np.linalg.norm): the numpy/BLAS
    # reduction differs in low bits across numpy builds, which broke cross-version
    # reproducibility. fsum over the Python-float weights is bit-stable everywhere, and the
    # remaining float32 division is element-wise IEEE — so the vector is deterministic.
    norm = math.sqrt(math.fsum(w * w for w in counts.values()))
    vec = np.zeros(DIM, dtype=np.float32)
    if norm > 0.0:
        for bucket, weight in counts.items():
            vec[bucket] = weight / norm
    else:
        for bucket, weight in counts.items():
            vec[bucket] = weight
    return vec


def embed_texts(texts: "list[str]") -> "list[np.ndarray]":
    return [embed_text(t) for t in texts]


def embed_query(text: str) -> np.ndarray:
    return embed_text(text)


__all__ = ["BACKEND_ID", "DIM", "embed_text", "embed_texts", "embed_query"]
