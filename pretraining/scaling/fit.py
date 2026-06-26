# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Power-law scaling-law fitting in pure Python (no numpy/scipy).

Fits ``L(x) = E + A · x^(-p)`` — the canonical Chinchilla-style form, where ``x`` is
data size ``D`` (or parameter count ``N``). Two modes:

* ``fit_with_floor`` — ``E`` is known (the analytic source entropy). Linear-regress
  ``log(L - E)`` on ``log(x)`` for an exact least-squares ``(A, p)``.
* ``fit_free_floor`` — ``E`` is unknown; ternary-search ``E`` to minimize the log-space
  residual, fitting ``(A, p)`` at each candidate. The recovered ``E`` is then *checked*
  against the analytic floor — that check is the study's honesty test.

``predict`` extrapolates the fitted law; ``r_squared`` reports fit quality in log space.
All deterministic and dependency-free.
"""
from __future__ import annotations

import math
from typing import Any


def _linreg(xs: "list[float]", ys: "list[float]") -> "tuple[float, float, float]":
    """Ordinary least squares y = a + b·x. Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


def fit_with_floor(xs: "list[float]", losses: "list[float]", floor: float) -> "dict[str, Any]":
    """Fit (A, p) for L = floor + A·x^-p, given a known irreducible ``floor``."""
    lx, ly = [], []
    for x, l in zip(xs, losses):
        excess = l - floor
        if x > 0 and excess > 0:
            lx.append(math.log(x))
            ly.append(math.log(excess))
    a, b, r2 = _linreg(lx, ly)
    return {"E": floor, "A": math.exp(a), "p": -b, "r2_logspace": r2,
            "n_points": len(lx), "floor_mode": "known"}


def fit_free_floor(xs: "list[float]", losses: "list[float]") -> "dict[str, Any]":
    """Fit (E, A, p) with E unknown via ternary search on the log-space residual."""
    lo = 0.0
    hi = min(losses) - 1e-6
    if hi <= lo:
        return {"E": float("nan"), "A": float("nan"), "p": float("nan"),
                "r2_logspace": float("nan"), "n_points": 0, "floor_mode": "free"}

    def residual(E: float) -> float:
        lx, ly = [], []
        for x, l in zip(xs, losses):
            ex = l - E
            if x > 0 and ex > 0:
                lx.append(math.log(x))
                ly.append(math.log(ex))
        if len(lx) < 2:
            return float("inf")
        a, b, _ = _linreg(lx, ly)
        return sum((y - (a + b * xx)) ** 2 for xx, y in zip(lx, ly))

    for _ in range(100):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if residual(m1) < residual(m2):
            hi = m2
        else:
            lo = m1
    E = (lo + hi) / 2
    fit = fit_with_floor(xs, losses, E)
    fit["floor_mode"] = "free"
    return fit


def predict(fit: "dict[str, Any]", x: float) -> float:
    """Predicted loss at ``x`` under the fitted law."""
    return fit["E"] + fit["A"] * (x ** (-fit["p"]))


__all__ = ["fit_with_floor", "fit_free_floor", "predict"]
