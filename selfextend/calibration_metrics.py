# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Trustworthy uncertainty everywhere: ECE + Brier over (confidence, correct) pairs.

A verdict's confidence is only useful if it tracks correctness. These are the
standard scores for that — Expected Calibration Error (binned gap between confidence
and accuracy) and the Brier score (mean squared error of probabilistic predictions).
Lower is better for both. Deterministic, dependency-free.
"""

from __future__ import annotations


def expected_calibration_error(pairs: "list[tuple[float, bool]]", *, bins: int = 10) -> float:
    """ECE: sum over confidence bins of (bin weight) * |avg confidence - accuracy|."""
    if not pairs:
        return 0.0
    buckets: list = [[] for _ in range(bins)]
    for conf, correct in pairs:
        idx = min(bins - 1, max(0, int(conf * bins)))
        buckets[idx].append((conf, bool(correct)))
    n = len(pairs)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_conf - acc)
    return round(ece, 4)


def brier_score(pairs: "list[tuple[float, bool]]") -> float:
    """Mean (confidence - outcome)^2."""
    if not pairs:
        return 0.0
    return round(sum((conf - (1.0 if ok else 0.0)) ** 2 for conf, ok in pairs) / len(pairs), 4)
