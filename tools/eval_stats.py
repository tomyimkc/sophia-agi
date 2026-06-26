#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measurement-instrument statistics for the Sophia evaluation contract (see
agi-proof/measurement-thesis.md). Pure-Python / stdlib only so it runs anywhere (CPU pod,
CI, dev box). Implements the runnable pillars:

  * Pillar 1 — uncertainty: fixed-n normal CI + paired percentile bootstrap.
  * Pillar 2 — power: required N for a target Minimum Detectable Effect (MDE).
  * Pillar 4 — ANYTIME-VALID inference: a time-uniform confidence sequence (Robbins normal
    mixture / Howard et al. 2021) whose coverage holds no matter how many times you peek or
    when you stop — the correct tool for an iterate-and-look workflow.
  * paired McNemar test for two adapters scored on the same items.

These are deliberately conservative (sub-Gaussian proxies) — they UNDER-claim rather than
over-claim, which is the point.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

# Standard-normal quantile (Acklam's rational approximation; good to ~1e-9).
_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]


def z_quantile(p: float) -> float:
    """Inverse standard-normal CDF."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0,1)")
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p <= 1 - pl:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
               (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
           ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)


def required_n_for_mde(mde: float, *, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8,
                       paired_rho: float = 0.0) -> int:
    """PILLAR 2. Smallest N whose test can detect an effect of size `mde` (in accuracy points)
    with `power`, at two-sided level `alpha`. Conservative one-sample-of-differences model:
    variance proxy of a per-item correctness difference is ~2*p0*(1-p0)*(1-paired_rho).
    Use paired_rho>0 if base/adapter correctness is positively correlated per item (it usually
    is — same items), which LOWERS the required N."""
    if mde <= 0:
        raise ValueError("mde must be > 0")
    za, zb = z_quantile(1 - alpha / 2), z_quantile(power)
    var = max(1e-6, 2 * p0 * (1 - p0) * (1 - paired_rho))
    n = ((za + zb) ** 2) * var / (mde ** 2)
    return int(math.ceil(n))


def mde_at_n(n: int, *, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> float:
    """Inverse of required_n_for_mde: the smallest effect an N-item probe can resolve. A probe
    whose mde_at_n exceeds the decision threshold is UNDERPOWERED by construction (the N=34 bug)."""
    za, zb = z_quantile(1 - alpha / 2), z_quantile(power)
    var = 2 * p0 * (1 - p0)
    return (za + zb) * math.sqrt(var / max(1, n))


def fixed_n_ci_mean(values: Sequence[float], alpha: float = 0.05) -> "list[float]":
    """PILLAR 1 (point-in-time). Normal-approx CI for a mean — valid only at a PRE-SPECIFIED n."""
    n = len(values)
    if n == 0:
        return [None, None]
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / max(1, n - 1)
    rad = z_quantile(1 - alpha / 2) * math.sqrt(var / n)
    return [round(m - rad, 4), round(m + rad, 4)]


def bootstrap_ci_paired(diffs: Sequence[float], alpha: float = 0.05, iters: int = 4000,
                        seed: int = 0) -> "list[float]":
    """PILLAR 1. Paired percentile-bootstrap CI for the mean difference (fixed-n)."""
    n = len(diffs)
    if n == 0:
        return [None, None]
    rnd = random.Random(seed)
    boot = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rnd.randrange(n)]
        boot.append(s / n)
    boot.sort()
    return [round(boot[int((alpha / 2) * iters)], 4), round(boot[int((1 - alpha / 2) * iters)], 4)]


def confidence_sequence_mean(values: Sequence[float], alpha: float = 0.05, *, sigma: float = None,
                             n_ref: int = 50) -> "list[float]":
    """PILLAR 4 — ANYTIME-VALID. Time-uniform confidence sequence for a mean via the Robbins
    normal-mixture boundary (Howard, Ramdas, McAuliffe & Sekhon 2021). The interval covers the
    true mean SIMULTANEOUSLY for all n with prob >= 1-alpha, so it stays valid under optional
    stopping / repeated peeking — unlike fixed-n CIs. `sigma` is the (sub-Gaussian) scale; if
    None it is estimated EMPIRICALLY from the sample std (a practical empirical-mixture CS — for
    strictly bounded data the empirical-Bernstein CS is tighter still, but this already uses the
    DATA's variance instead of the worst-case range). `n_ref` tunes where the interval is tightest."""
    n = len(values)
    if n == 0:
        return [None, None]
    m = sum(values) / n
    if sigma is None:
        var = sum((v - m) ** 2 for v in values) / max(1, n - 1)
        sigma = math.sqrt(var) + 1e-9
    rho2 = 1.0 / max(1, n_ref)            # mixing variance (tuning); any rho2>0 keeps validity
    # radius_n = sigma * sqrt( 2 (n*rho2 + 1) / (n^2 rho2) * log( sqrt(n*rho2 + 1)/alpha ) )
    nr = n * rho2 + 1.0
    rad = sigma * math.sqrt((2.0 * nr / (n * n * rho2)) * math.log(math.sqrt(nr) / alpha))
    return [round(m - rad, 4), round(m + rad, 4)]


def mcnemar(base_correct: Sequence[int], adapter_correct: Sequence[int]) -> dict:
    """Paired McNemar test for two adapters on the SAME items. b = base-right/adapter-wrong,
    c = base-wrong/adapter-right. Returns the discordant counts + a chi-square (cc) p-value."""
    b = sum(1 for x, y in zip(base_correct, adapter_correct) if x and not y)
    c = sum(1 for x, y in zip(base_correct, adapter_correct) if y and not x)
    if b + c == 0:
        return {"b": 0, "c": 0, "stat": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected
    # chi-square(df=1) survival = 2*(1-Phi(sqrt(stat)))
    p = 2 * (1 - _phi(math.sqrt(stat)))
    return {"b": b, "c": c, "stat": round(stat, 4), "p": round(min(1.0, p), 4)}


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


if __name__ == "__main__":
    # self-test / demo
    print("required N for MDE 0.05 @ p0=0.5:", required_n_for_mde(0.05))
    print("MDE resolvable at N=34:", round(mde_at_n(34), 3), "| at N=70:", round(mde_at_n(70), 3))
    diffs = [(-1 if i < 1 else 0) for i in range(70)]  # one item flips negative out of 70
    print("Δ=-1/70:", round(sum(diffs) / len(diffs), 4),
          "| boot CI:", bootstrap_ci_paired(diffs),
          "| anytime CS:", confidence_sequence_mean(diffs))
