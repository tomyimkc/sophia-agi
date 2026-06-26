# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Selective-prediction metrics: risk-coverage, AURC, matched-coverage fabrication.

An item is (confidence in [0,1], fabricated in {0,1}). At a coverage c the predictor
answers the most-confident ceil(c*N) items; selective fabrication is the fabrication
rate over those answered items. This neutralizes the "it just abstains" rebuttal: arms
are compared at the SAME coverage, so abstention is no longer free.
"""
from __future__ import annotations

import math
import random
import statistics
from typing import Sequence

Item = tuple[float, int]


def _rank(items: Sequence[Item]) -> list[int]:
    """Indices sorted by confidence desc, index asc as a deterministic tiebreak."""
    return sorted(range(len(items)), key=lambda i: (-float(items[i][0]), i))


def coverage_fabrication_at(items: Sequence[Item], coverage: float) -> float:
    n = len(items)
    if n == 0:
        return 0.0
    k = math.ceil(max(0.0, min(1.0, coverage)) * n)
    if k <= 0:
        return 0.0
    order = _rank(items)[:k]
    return statistics.fmean(int(items[i][1]) for i in order)


def aurc(items: Sequence[Item]) -> float:
    """Discrete AURC: mean selective fabrication over prefixes k=1..N. Lower is better."""
    n = len(items)
    if n == 0:
        return 0.0
    order = _rank(items)
    risks, fab = [], 0
    for k, i in enumerate(order, start=1):
        fab += int(items[i][1])
        risks.append(fab / k)
    return statistics.fmean(risks)


def paired_aurc_delta_ci(
    raw: Sequence[Item],
    full: Sequence[Item],
    *,
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> list[float]:
    """Percentile CI for aurc(raw) - aurc(full) under a paired item bootstrap."""
    if len(raw) != len(full):
        raise ValueError("paired AURC bootstrap needs equal-length arms")
    n = len(raw)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(aurc([raw[i] for i in idx]) - aurc([full[i] for i in idx]))
    deltas.sort()
    lo_i = max(0, min(n_boot - 1, int((alpha / 2) * n_boot)))
    hi_i = max(0, min(n_boot - 1, int((1 - alpha / 2) * n_boot) - 1))
    return [round(float(deltas[lo_i]), 6), round(float(deltas[hi_i]), 6)]
