# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cluster observability — time-series summaries, tail/jitter, straggler analysis.

This is the measurement half of the JD's "构建集群可观测性体系 … 分析性能抖动、长尾延迟
以及性能不均". Pure stdlib (no numpy) so it runs in CI. Two things it gives you:

  1. summarize(values)          -> mean / p50 / p90 / p99 / max / cv (jitter)
  2. straggler_report(per_rank) -> which rank is the long pole, tail ratio (p99/p50),
                                   skew across ranks, and the slowdown a single straggler
                                   imposes on a synchronous (all-reduce) step.

A synchronous training step runs at the speed of its slowest rank, so a 1% straggler
rate at the step level compounds across thousands of steps — exactly the systemic
"performance unevenness" the role is asked to hunt down.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0,100]. Empty -> 0.0."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    rank = (q / 100.0) * (len(xs) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return xs[lo]
    frac = rank - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


@dataclass
class Summary:
    n: int
    mean: float
    p50: float
    p90: float
    p99: float
    max: float
    cv: float          # coefficient of variation = stdev/mean — a unit-free jitter score

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def summarize(values: list[float]) -> Summary:
    m = _mean(values)
    sd = _stdev(values)
    return Summary(
        n=len(values),
        mean=m,
        p50=percentile(values, 50),
        p90=percentile(values, 90),
        p99=percentile(values, 99),
        max=max(values) if values else 0.0,
        cv=(sd / m) if m else 0.0,
    )


@dataclass
class StragglerReport:
    n_ranks: int
    slowest_rank: int
    fastest_rank: int
    tail_ratio: float          # p99 / p50 of per-rank step time
    skew: float                # (max - min) / mean across ranks
    step_slowdown: float       # slowest / mean — the all-reduce tax per synchronous step
    summary: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tail_ratio"] = round(self.tail_ratio, 4)
        d["skew"] = round(self.skew, 4)
        d["step_slowdown"] = round(self.step_slowdown, 4)
        return d


def straggler_report(per_rank_step_s: list[float]) -> StragglerReport:
    """Given each rank's mean step time, quantify the long pole of a sync step.

    `step_slowdown` is the multiplier a synchronous (barrier/all-reduce) step pays:
    every rank waits for the slowest, so useful throughput = mean/slowest of peak.
    """
    if not per_rank_step_s:
        raise ValueError("per_rank_step_s is empty")
    s = summarize(per_rank_step_s)
    slowest = max(range(len(per_rank_step_s)), key=lambda i: per_rank_step_s[i])
    fastest = min(range(len(per_rank_step_s)), key=lambda i: per_rank_step_s[i])
    mean = s.mean or 1.0
    return StragglerReport(
        n_ranks=len(per_rank_step_s),
        slowest_rank=slowest,
        fastest_rank=fastest,
        tail_ratio=(s.p99 / s.p50) if s.p50 else 1.0,
        skew=(max(per_rank_step_s) - min(per_rank_step_s)) / mean,
        step_slowdown=max(per_rank_step_s) / mean,
        summary=s.as_dict(),
    )
