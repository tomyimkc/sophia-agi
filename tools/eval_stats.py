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


def mde_at_n(n: int, *, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8,
             paired_rho: float = 0.0) -> float:
    """Inverse of required_n_for_mde: the smallest effect an N-item probe can resolve. A probe
    whose mde_at_n exceeds the decision threshold is UNDERPOWERED by construction (the N=34 bug).
    `paired_rho>0` reflects per-item base/adapter correlation (same items), which LOWERS the MDE;
    leave it 0 for the conservative worst-case (the default used by the claim gate)."""
    za, zb = z_quantile(1 - alpha / 2), z_quantile(power)
    var = 2 * p0 * (1 - p0) * (1 - paired_rho)
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


def verdict_or_underpowered(delta: float, n: int, *, tolerance: float = None,
                            p0: float = 0.5, alpha: float = 0.05, power: float = 0.8,
                            paired_rho: float = 0.0, up: str = "improves", down: str = "regresses",
                            flat: str = "no-change") -> dict:
    """PILLAR 2/3 GUARD. Turn a raw delta into a verdict ONLY when the probe can resolve it.

    Refuses to emit a directional verdict word when the probe's MDE at this N exceeds the
    effect being judged — the failure mode that produced the spurious N=34 '-0.118 forgetting'
    read. The decision threshold is `tolerance` (default: the magnitude of the delta itself, so
    a probe must at least be able to see an effect of the size observed). Returns a dict carrying
    the chosen word, the MDE, and a self-describing `mde` field so a committed eval JSON is
    auditable. Callers should print/serialize the whole dict — never a bare verdict string."""
    mde = round(mde_at_n(max(1, n), p0=p0, alpha=alpha, power=power, paired_rho=paired_rho), 4)
    tol = abs(delta) if tolerance is None else abs(tolerance)
    powered = mde <= tol + 1e-9
    if not powered:
        word, note = "underpowered", (f"MDE {mde} > tolerance {round(tol,4)} at N={n}: the probe "
                                      f"CANNOT resolve this effect — grow N before claiming a direction")
    elif abs(delta) < mde:
        # probe is powered for a tolerance-sized effect, but THIS delta is below the MDE -> its
        # sign is not statistically resolvable. Honest verdict is no-change, not a tiny direction.
        word, note = flat, (f"|delta| {round(abs(delta),4)} < MDE {mde}: sign not resolvable "
                            f"(powered for a {round(tol,4)} effect; none of that size observed)")
    else:
        word, note = (up if delta > 0 else down), f"MDE {mde} <= tolerance {round(tol,4)}: resolvable"
    return {"verdict": word, "powered": powered, "delta": round(delta, 4), "n": n,
            "mde": mde, "tolerance": round(tol, 4), "note": note}


def confidence_sequence_from_summary(mean: float, n: int, sigma: float, alpha: float = 0.05,
                                     n_ref: int = 50) -> "list[float]":
    """PILLAR 4 — anytime-valid interval from SUMMARY stats (mean, n, sigma) when per-item data
    isn't retained. Same Robbins normal-mixture boundary as confidence_sequence_mean; use this to
    retrofit a peeking-robust interval onto a result whose only stored uncertainty is a fixed-n CI.
    `sigma` can be backed out of a fixed-n CI half-width h via sigma = h*sqrt(n)/z_quantile(1-alpha/2)."""
    if n <= 0 or sigma <= 0:
        return [None, None]
    rho2 = 1.0 / max(1, n_ref)
    nr = n * rho2 + 1.0
    rad = sigma * math.sqrt((2.0 * nr / (n * n * rho2)) * math.log(math.sqrt(nr) / alpha))
    return [round(mean - rad, 4), round(mean + rad, 4)]


def sigma_from_ci(ci_lo: float, ci_hi: float, n: int, alpha: float = 0.05) -> float:
    """Back out the per-item sub-Gaussian sigma implied by a fixed-n normal/bootstrap CI."""
    h = (ci_hi - ci_lo) / 2.0
    return h * math.sqrt(max(1, n)) / z_quantile(1 - alpha / 2)


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
    # power-or-no-verdict guard: the N=34 '-0.118 forgetting' read should NOT get a verdict word
    print("verdict@N=34 Δ-0.118 tol=0.05:", verdict_or_underpowered(-0.118, 34, tolerance=0.05))
    print("verdict@N=970 Δ-0.001 tol=0.05 (paired rho=0.5):",
          verdict_or_underpowered(-0.001, 970, tolerance=0.05, paired_rho=0.5))
