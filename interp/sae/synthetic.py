# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic synthetic activations for the offline SAE core (no numpy).

A planted sparse dictionary: each "activation" is a positive sparse combination of
a few fixed unit feature directions plus small noise — exactly the structure a
TopK SAE should recover. Used by the M0 mock CLI and the unit tests so the SAE is
validated against a *known* ground truth before any GPU harvest.
"""
from __future__ import annotations

import math
import random


def _unit(v: "list[float]") -> "list[float]":
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else v


def feature_dictionary(d_in: int, n_features: int, *, seed: int = 0) -> "list[list[float]]":
    """`n_features` fixed unit directions in R^{d_in} (the ground-truth dictionary)."""
    rng = random.Random(seed)
    return [_unit([rng.gauss(0.0, 1.0) for _ in range(d_in)]) for _ in range(n_features)]


def planted_activations(
    n: int,
    d_in: int = 12,
    n_features: int = 6,
    k_true: int = 2,
    *,
    seed: int = 0,
    noise: float = 0.02,
) -> "tuple[list[list[float]], list[list[float]]]":
    """Return (X, dictionary). Each x = Σ_{f∈S} c_f · D[f] + ε, |S| = k_true,
    c_f ∼ U[0.5,1.5], ε ∼ N(0, noise²). Deterministic for a fixed seed."""
    rng = random.Random(seed)
    D = feature_dictionary(d_in, n_features, seed=seed)
    X = []
    for _ in range(n):
        active = rng.sample(range(n_features), k_true)
        x = [0.0] * d_in
        for f in active:
            c = rng.uniform(0.5, 1.5)
            df = D[f]
            for i in range(d_in):
                x[i] += c * df[i]
        if noise > 0.0:
            for i in range(d_in):
                x[i] += rng.gauss(0.0, noise)
        X.append(x)
    return X, D
