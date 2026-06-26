# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic, dependency-free lexical embedding for offline vector retrieval.

This is **NOT** a learned semantic embedding. It hashes word unigrams and
character n-grams into a fixed-dimension vector with sublinear term-frequency
weighting and L2 normalisation, then ranks by cosine similarity. Its value over
the plain keyword token-overlap scorer (``agent.retrieval._score``) is sub-word
matching — morphological variants ("philosophy" vs "philosophical") and partial
overlaps that an exact token-set intersection misses.

Why it exists: the learned-embedding path in ``agent.vector_store`` /
``agent.rag_embed`` needs a model + API (GOOGLE_API_KEY / Vertex) and a committed
``embeddings.npz``. When those are absent — airgapped runs, CI, this offline
harness — retrieval silently degraded to keyword-only. This module is the
reproducible offline middle tier: no numpy, no model, no network. Reproducible
across processes because it uses ``hashlib`` (stable), never the salted built-in
``hash()``. A learned backend remains pluggable and takes precedence when present.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

DIM = 256
NGRAM = 3

_WORD = re.compile(r"[a-z0-9一-鿿]+")


def _features(text: str) -> list[str]:
    """Word unigrams plus padded character tri-grams of each word."""
    feats: list[str] = []
    for tok in _WORD.findall(text.lower()):
        feats.append(tok)
        padded = f"^{tok}$"
        if len(padded) <= NGRAM:
            feats.append(padded)
            continue
        for i in range(len(padded) - NGRAM + 1):
            feats.append(padded[i : i + NGRAM])
    return feats


def embed(text: str, *, dim: int = DIM) -> list[float]:
    """Return an L2-normalised signed-hashing vector for ``text`` (numpy-free)."""
    vec = [0.0] * dim
    counts = Counter(_features(text))
    for feat, count in counts.items():
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "big") % dim
        sign = 1.0 if (digest[0] & 1) == 0 else -1.0  # signed hashing limits collision bias
        vec[idx] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vec))
    if norm > 0.0:
        vec = [value / norm for value in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine for two equal-length L2-normalised vectors, clamped to [-1, 1]."""
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def rank(query: str, docs: Iterable[tuple[str, str]], *, top_k: int = 8, dim: int = DIM) -> list[tuple[str, float]]:
    """Rank ``(key, text)`` docs by cosine to ``query``; returns ``(key, score)`` desc.

    Zero-similarity docs are dropped (parity with the keyword scorer's score>0 filter).
    """
    q = embed(query, dim=dim)
    scored: list[tuple[str, float]] = []
    for key, text in docs:
        score = cosine(q, embed(text, dim=dim))
        if score > 0.0:
            scored.append((key, score))
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return scored[:top_k]
