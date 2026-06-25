# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Loss-of-plasticity probe (Hurdle 4 — don't let the continual loop degrade).

The 2025 continual-learning literature pins slow degradation in repeatedly-trained
networks on measurable correlates: **feature/weight-rank collapse**, **dormant units**,
and **growing weight norms** (e.g. arXiv 2509.22335 spectral collapse; 2404.00781).
This module computes those correlates from a weight (or activation) matrix so a
generational run can carry an *early-warning* signal alongside its accuracy curve.

It is a DIAGNOSTIC, not a gate: a rising accuracy curve that is simultaneously losing
stable rank / gaining dead units is the signature of a loop about to plateau-then-rot.
Pure Python (no numpy): spectral norm via deterministic power iteration, so it runs in
CI and on any host.

Honest bound: these are *correlates* of plasticity loss, not a proof of it. They flag
where to look; the fix (shrink-and-perturb / L2-init / continual-backprop between
generations) and the real measurement run on hardware.
"""

from __future__ import annotations

import math
from typing import Any

Matrix = "list[list[float]]"


def _l2(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def frobenius_norm(m: Matrix) -> float:
    return math.sqrt(sum(x * x for row in m for x in row))


def spectral_norm(m: Matrix, *, iters: int = 200, tol: float = 1e-12) -> float:
    """Largest singular value via power iteration on ``MᵀM`` (deterministic start).

    Returns 0.0 for an empty/zero matrix. The all-ones start vector is deterministic
    (no RNG, so results are reproducible and resume-safe); it converges to the top
    singular value for any matrix whose leading singular vector is not orthogonal to it,
    which holds for the dense weight matrices this probe targets.
    """
    if not m or not m[0]:
        return 0.0
    ncols = len(m[0])
    v = [1.0] * ncols
    norm_v = _l2(v)
    if norm_v == 0:
        return 0.0
    v = [x / norm_v for x in v]
    sigma = 0.0
    for _ in range(iters):
        # w = M v
        w = [sum(row[j] * v[j] for j in range(ncols)) for row in m]
        # u = Mᵀ w
        u = [sum(m[i][j] * w[i] for i in range(len(m))) for j in range(ncols)]
        norm_u = _l2(u)
        if norm_u == 0:
            return 0.0
        v = [x / norm_u for x in u]
        new_sigma = math.sqrt(norm_u)
        if abs(new_sigma - sigma) <= tol:
            sigma = new_sigma
            break
        sigma = new_sigma
    return sigma


def stable_rank(m: Matrix) -> float:
    """‖M‖_F² / ‖M‖₂²  — a smooth, SVD-free proxy for effective rank in [0, min(r,c)].

    Falls toward 1.0 as the matrix collapses onto a single direction (the rank-collapse
    correlate of plasticity loss); near min(rows, cols) for a well-conditioned matrix.
    """
    spec = spectral_norm(m)
    if spec == 0:
        return 0.0
    fro = frobenius_norm(m)
    return round((fro * fro) / (spec * spec), 4)


def dead_unit_fraction(m: Matrix, *, eps: float = 1e-6, axis: int = 0) -> float:
    """Fraction of units whose weight vector is ~zero (dormant-neuron correlate).

    ``axis=0`` treats each row as a unit; ``axis=1`` each column.
    """
    if not m or not m[0]:
        return 0.0
    if axis == 0:
        units = m
    else:
        units = [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]
    dead = sum(1 for u in units if _l2(u) < eps)
    return round(dead / len(units), 4)


def plasticity_report(m: Matrix, *, name: str = "weight", dead_eps: float = 1e-6) -> dict[str, Any]:
    """Per-matrix plasticity correlates: stable rank, dead-unit fraction, weight norm."""
    rows = len(m)
    cols = len(m[0]) if m else 0
    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "maxRank": min(rows, cols),
        "frobeniusNorm": round(frobenius_norm(m), 6),
        "spectralNorm": round(spectral_norm(m), 6),
        "stableRank": stable_rank(m),
        "deadUnitFraction": dead_unit_fraction(m, eps=dead_eps),
    }


def watch_generations(
    reports: list[dict[str, Any]],
    *,
    rank_drop_frac: float = 0.15,
    norm_growth_ratio: float = 2.0,
    dead_rise: float = 0.10,
) -> dict[str, Any]:
    """Compare per-generation plasticity reports and flag degradation correlates.

    Flags (early-warning, not a verdict on capability):
      - ``stableRankFalling``  : stable rank drops > ``rank_drop_frac`` vs the first gen
      - ``weightNormGrowing``  : Frobenius norm grows > ``norm_growth_ratio`` × the first
      - ``deadUnitsRising``    : dead-unit fraction rises > ``dead_rise`` vs the first

    ``reports`` is an ordered list of ``plasticity_report`` dicts (one per generation).
    """
    if len(reports) < 2:
        return {"watched": len(reports), "flags": {}, "verdict": "insufficient-generations",
                "boundary": "Correlates of plasticity loss, not a proof; diagnostic only."}

    first, last = reports[0], reports[-1]
    sr0, srN = first["stableRank"], last["stableRank"]
    fn0, fnN = first["frobeniusNorm"], last["frobeniusNorm"]
    du0, duN = first["deadUnitFraction"], last["deadUnitFraction"]

    flags = {
        "stableRankFalling": bool(sr0 > 0 and (sr0 - srN) / sr0 > rank_drop_frac),
        "weightNormGrowing": bool(fn0 > 0 and fnN / fn0 > norm_growth_ratio),
        "deadUnitsRising": bool((duN - du0) > dead_rise),
    }
    degrading = any(flags.values())
    return {
        "schema": "sophia.plasticity_watch.v1",
        "watched": len(reports),
        "first": {"stableRank": sr0, "frobeniusNorm": fn0, "deadUnitFraction": du0},
        "last": {"stableRank": srN, "frobeniusNorm": fnN, "deadUnitFraction": duN},
        "flags": flags,
        "verdict": "degrading-plasticity-warning" if degrading else "plasticity-stable",
        "boundary": ("Correlates of plasticity loss (rank collapse, dead units, weight-norm "
                     "growth), not a proof. Mitigation (shrink-and-perturb / L2-init / "
                     "continual-backprop) runs at the retrain step on hardware."),
    }
