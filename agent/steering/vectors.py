# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pure-stdlib residual-stream vector math (no torch, no numpy).

A Vector is a list[float]. The real path (hooks.py) converts torch hidden states
to plain lists before calling these, so this module stays CI-testable.
"""
from __future__ import annotations

import math
import random

Vector = list  # list[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def scale(a: Vector, s: float) -> Vector:
    return [x * s for x in a]


def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def mean_vectors(vs: "list[Vector]") -> Vector:
    if not vs:
        return []
    n = len(vs)
    dim = len(vs[0])
    acc = [0.0] * dim
    for v in vs:
        for i in range(dim):
            acc[i] += v[i]
    return [x / n for x in acc]


def diff_of_means(pos: "list[Vector]", neg: "list[Vector]") -> Vector:
    """CAA Eq. 1: mean(positive activations) − mean(negative activations)."""
    return sub(mean_vectors(pos), mean_vectors(neg))


def normalize(v: Vector) -> Vector:
    n = norm(v)
    return v[:] if n == 0.0 else scale(v, 1.0 / n)


def cosine(a: Vector, b: Vector) -> float:
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def mock_vector(dim: int, seed: int) -> Vector:
    """Deterministic seeded unit vector — the offline extractor stand-in."""
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    return normalize(v)
