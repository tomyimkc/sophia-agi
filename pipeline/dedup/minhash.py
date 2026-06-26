# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""MinHash + LSH near-duplicate detection (Phase 2), pure stdlib.

The JD names "MinHash 去重" as a core cleaning step. This is a from-scratch, dependency-free
implementation (no ``datasketch``) so it runs under ``SOPHIA_PROFILE=airgap`` and in CI with
only the stdlib:

  - **shingling** — word k-grams (falls back to char n-grams for very short texts);
  - **MinHash** — ``num_perm`` permutations ``(a*h + b) mod (2^61 - 1)`` over a stable
    ``blake2b`` shingle hash (NOT the salted builtin ``hash``, so signatures are identical
    across runs / processes / platforms — same discipline as ``agent.rag_local_embed``);
  - **LSH banding** — split the signature into ``bands`` bands of ``num_perm // bands`` rows;
    documents sharing a band bucket are candidate pairs, refined by an estimated-Jaccard
    threshold and unioned into clusters.

Deterministic: same texts -> same signatures -> same clusters.
"""

from __future__ import annotations

import hashlib
import re

_MERSENNE_61 = (1 << 61) - 1
_MAX_HASH = (1 << 64) - 1
_WORD_RE = re.compile(r"[a-z0-9一-鿿]+")


def _stable_hash(s: str) -> int:
    """A stable 64-bit hash of ``s`` (blake2b, not the salted builtin ``hash``)."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def _permutation_params(num_perm: int) -> list[tuple[int, int]]:
    """Deterministic (a, b) coefficients for ``num_perm`` hash permutations."""
    params: list[tuple[int, int]] = []
    for i in range(num_perm):
        a = _stable_hash(f"minhash:a:{i}") % (_MERSENNE_61 - 1) + 1
        b = _stable_hash(f"minhash:b:{i}") % _MERSENNE_61
        params.append((a, b))
    return params


def shingles(text: str, k: int = 5) -> set[str]:
    """Word k-gram shingles; falls back to char trigrams when the text is too short."""
    tokens = _WORD_RE.findall(text.lower())
    if len(tokens) >= k:
        return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}
    if tokens:
        joined = " ".join(tokens)
        if len(joined) >= 3:
            return {joined[i : i + 3] for i in range(len(joined) - 2)}
        return {joined}
    return set()


def signature(text: str, *, num_perm: int = 64, _params=None) -> tuple[int, ...]:
    """MinHash signature of ``text`` as a tuple of ``num_perm`` ints."""
    params = _params if _params is not None else _permutation_params(num_perm)
    sh = shingles(text)
    if not sh:
        return tuple([_MAX_HASH] * num_perm)
    hashes = [_stable_hash(s) for s in sh]
    sig = []
    for a, b in params:
        sig.append(min(((a * h + b) % _MERSENNE_61) for h in hashes))
    return tuple(sig)


def jaccard_estimate(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
    """Estimated Jaccard similarity = fraction of matching signature positions."""
    if not sig_a or len(sig_a) != len(sig_b):
        return 0.0
    matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return matches / len(sig_a)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster(
    texts, *, threshold: float = 0.8, num_perm: int = 64, bands: int = 16
) -> tuple[list[int], list[tuple[int, ...]]]:
    """Cluster near-duplicate texts.

    Returns ``(cluster_ids, signatures)`` aligned to input order. ``cluster_ids[i]`` is the
    smallest index in document ``i``'s similarity cluster (so singletons map to themselves).
    """
    texts = list(texts)
    if num_perm % bands != 0:
        raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands})")
    rows = num_perm // bands
    params = _permutation_params(num_perm)
    sigs = [signature(t, num_perm=num_perm, _params=params) for t in texts]

    # LSH: bucket by each band; same bucket -> candidate pair.
    uf = _UnionFind(len(texts))
    buckets: dict[tuple, list[int]] = {}
    for i, sig in enumerate(sigs):
        for band in range(bands):
            chunk = sig[band * rows : (band + 1) * rows]
            key = (band, _stable_hash(repr(chunk)))
            buckets.setdefault(key, []).append(i)

    for members in buckets.values():
        if len(members) < 2:
            continue
        first = members[0]
        for j in members[1:]:
            if jaccard_estimate(sigs[first], sigs[j]) >= threshold:
                uf.union(first, j)

    cluster_ids = [uf.find(i) for i in range(len(texts))]
    return cluster_ids, sigs


__all__ = ["shingles", "signature", "jaccard_estimate", "cluster"]
