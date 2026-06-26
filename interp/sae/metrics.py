# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SAE evaluation metrics — pure stdlib (no numpy/torch).

The metrics the roadmap (§6) pre-registers for SAE quality: L0, reconstruction
(FVU), cross-entropy-loss recovered, dead-feature %, and a feature-density
histogram — plus bootstrap CIs reusing the repo's `_ci` so intervals match the
rest of the codebase. References: Bricken et al. 2023 (density/dead features),
Gao et al. 2024 (TopK eval), Anthropic/EleutherAI SAEBench 2024 (standardized
suite). CE-loss-recovered is the metric that matters most: reconstruction can
look fine while the substituted forward pass collapses.
"""
from __future__ import annotations

import math
import random

from provenance_bench.aggregate import _ci  # shared CI helper (same as steering/stats)


def l0(codes: "list[list[float]]") -> float:
    """Mean number of nonzero (active) features per token."""
    if not codes:
        return 0.0
    return sum(sum(1 for v in row if v != 0.0) for row in codes) / len(codes)


def _col_means(X: "list[list[float]]") -> "list[float]":
    n = len(X)
    d = len(X[0])
    return [sum(X[r][j] for r in range(n)) / n for j in range(d)]


def fvu(X: "list[list[float]]", X_hat: "list[list[float]]") -> float:
    """Fraction of Variance Unexplained = Σ||x−x̂||² / Σ||x−mean||².

    0.0 = perfect reconstruction; 1.0 = no better than predicting the mean.
    """
    if not X:
        return 0.0
    mean = _col_means(X)
    num = 0.0
    den = 0.0
    for x, xh in zip(X, X_hat):
        for j in range(len(x)):
            num += (x[j] - xh[j]) ** 2
            den += (x[j] - mean[j]) ** 2
    if den <= 0.0:
        return 0.0
    return num / den


def explained_variance(X: "list[list[float]]", X_hat: "list[list[float]]") -> float:
    return 1.0 - fvu(X, X_hat)


def ce_loss_recovered(clean_ce: float, recon_ce: float, ablated_ce: float) -> float:
    """Fraction of the CE gap (mean-ablation → clean) recovered by the SAE.

    1.0 = substituting the SAE reconstruction preserves the model's loss exactly;
    0.0 = no better than ablating the activation. Pre-registered target ≥ ~0.9.
    """
    denom = ablated_ce - clean_ce
    if abs(denom) < 1e-12:
        return 0.0
    return (ablated_ce - recon_ce) / denom


def dead_feature_fraction(codes: "list[list[float]]", d_hidden: int) -> float:
    """Fraction of dictionary features that never fire over the eval set."""
    if d_hidden <= 0:
        return 0.0
    live = set()
    for row in codes:
        for h, v in enumerate(row):
            if v != 0.0:
                live.add(h)
    return (d_hidden - len(live)) / d_hidden


def feature_firing_rate(codes: "list[list[float]]", d_hidden: int) -> "list[float]":
    """Per-feature fraction of tokens on which it is active."""
    n = len(codes) or 1
    counts = [0] * d_hidden
    for row in codes:
        for h, v in enumerate(row):
            if v != 0.0:
                counts[h] += 1
    return [c / n for c in counts]


def density_histogram(
    codes: "list[list[float]]", d_hidden: int, *, n_bins: int = 10
) -> "dict":
    """log10(firing-rate) histogram (Bricken density plot). Features that never
    fire go in a separate 'dead' count; live features bucket over [−n_bins, 0]."""
    rates = feature_firing_rate(codes, d_hidden)
    dead = sum(1 for r in rates if r == 0.0)
    bins = [0] * n_bins  # bin i covers log10-rate in [-(n_bins-i), -(n_bins-i-1))
    for r in rates:
        if r == 0.0:
            continue
        lg = math.log10(r)  # in (-inf, 0]
        idx = min(n_bins - 1, max(0, int(n_bins + math.floor(lg))))
        bins[idx] += 1
    return {"dead": dead, "bins": bins, "nBins": n_bins, "nFeatures": d_hidden}


def bootstrap_ci(
    values: "list[float]", *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> "list[float]":
    """Bootstrap 95% CI of the mean of `values`, via the repo's `_ci` helper."""
    n = len(values)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(s) / n)
    return _ci(boot, alpha)
