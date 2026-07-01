#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rank-correlation statistics for the topology->truth axiom probe (additive to
`tools/eval_stats.py`; this module does NOT edit or shadow it).

Provides:
  * `spearman_rho(xs, ys)`      — Spearman rank correlation (ties -> average ranks).
  * `permutation_pvalue(...)`   — a deterministic, seeded permutation test for the
    monotone association between xs and ys, using |rho| as the statistic (two-sided by
    default). Stdlib only so it runs on any CPU box / CI.

Why a PERMUTATION p-value and not a parametric one: n is tiny (~20-30 labeled claims)
and the truth label is binary (0/1), so the asymptotic t-approximation for Spearman is
untrustworthy at this N. A permutation test makes NO distributional assumption — it just
asks how often a random relabelling of truth reaches an association at least as extreme
as observed. It is exact-in-the-limit-of-all-permutations and here uses a fixed, seeded
Monte-Carlo sample so the p-value is fully reproducible.

These are deliberately conservative and stdlib-only; they UNDER-claim rather than
over-claim, matching the repo's no-overclaim law.
"""
from __future__ import annotations

import random
from typing import Sequence


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Rank `values` ascending, assigning the average rank to ties (1-based)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # advance j over the run of equal values
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j (0-based) -> 1-based ranks (i+1)..(j+1); average them
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation of two equal-length sequences. Returns 0.0 if either is
    constant (undefined correlation -> treat as no linear association)."""
    n = len(xs)
    if n == 0 or n != len(ys):
        raise ValueError("xs and ys must be non-empty and equal length")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation coefficient in [-1, 1].

    Equal to the Pearson correlation of the average-rank transforms of the inputs, so it
    handles ties correctly. Returns 0.0 when either variable is constant.

    Raises ValueError if the inputs are empty or of unequal length.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have equal length")
    if len(xs) == 0:
        raise ValueError("xs and ys must be non-empty")
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    return _pearson(rx, ry)


def permutation_pvalue(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    iters: int = 10000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> dict:
    """Deterministic seeded permutation test for a monotone xs<->ys association.

    Statistic is `spearman_rho(xs, ys)`. Under the null (no association) the ys labels are
    exchangeable, so we permute ys `iters` times and count how often the permuted statistic
    is at least as extreme as the observed one.

    `alternative`:
      * "two-sided" (default): extreme means |rho_perm| >= |rho_obs|.
      * "greater":             rho_perm >= rho_obs (tests the axiom's directional claim
                               that topology confidence rises with truth).
      * "less":                rho_perm <= rho_obs.

    The p-value uses the +1 (add-one) small-sample correction:
        p = (1 + #{as-extreme}) / (1 + iters)
    which keeps p strictly positive and is the standard unbiased estimator (Phipson &
    Smyth 2010). Fully reproducible for a fixed `seed`.

    Returns a dict: {"rho", "p", "iters", "seed", "alternative", "count_as_extreme"}.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be one of: two-sided, greater, less")
    if iters < 1:
        raise ValueError("iters must be >= 1")
    rho_obs = spearman_rho(xs, ys)
    rng = random.Random(seed)
    ys_list = list(ys)
    count = 0
    for _ in range(iters):
        perm = ys_list[:]
        rng.shuffle(perm)
        rho_p = spearman_rho(xs, perm)
        if alternative == "two-sided":
            if abs(rho_p) >= abs(rho_obs) - 1e-12:
                count += 1
        elif alternative == "greater":
            if rho_p >= rho_obs - 1e-12:
                count += 1
        else:  # less
            if rho_p <= rho_obs + 1e-12:
                count += 1
    p = (1 + count) / (1 + iters)
    return {
        "rho": rho_obs,
        "p": p,
        "iters": iters,
        "seed": seed,
        "alternative": alternative,
        "count_as_extreme": count,
    }


__all__ = ["spearman_rho", "permutation_pvalue"]
