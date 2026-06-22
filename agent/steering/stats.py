"""Effect-size, bootstrap CI, kappa, and the pre-registered SSA verdict (pure stdlib).

Reuses provenance_bench's stdlib helpers verbatim — no new stats deps.
"""
from __future__ import annotations

import random
import statistics

from provenance_bench.aggregate import _ci, KAPPA_FLOOR
from provenance_bench.consensus import cohen_kappa  # re-exported for callers

# Pre-registered SSA thresholds — fixed before any run (spec §"Locked decisions").
SSA_THRESHOLDS = {
    "delta_point_min": 0.30,   # Δd point estimate floor
    "steered_d_min": 0.50,     # absolute residualized d floor
    "off_target_max": 0.20,    # off-target |d| null band
    "kappa_floor": KAPPA_FLOOR,  # 0.40
    "capability_eps": 0.05,    # ≤5% relative capability drop
    "coherence_floor": 75.0,
}


def cohen_d(a: "list[float]", b: "list[float]") -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = statistics.pvariance(a), statistics.pvariance(b)
    pooled = ((va + vb) / 2.0) ** 0.5
    if pooled == 0.0:
        return 0.0
    return (statistics.fmean(a) - statistics.fmean(b)) / pooled


def bootstrap_diff_ci(steer: "list[float]", base: "list[float]", *,
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> "list[float]":
    """Bootstrap CI of the paired difference (steer_i − base_i). steer/base are
    per-seed effect sizes already, paired by index. Returns _ci([..]) = [lo, hi]."""
    diffs = [s - b for s, b in zip(steer, base)]
    n = len(diffs)
    if n == 0:
        return [0.0, 0.0]
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot.append(statistics.fmean(sample))
    return _ci(boot, alpha)


def binarize_moved(scores: "list[float]", neutral: "list[float]") -> "list[int]":
    return [1 if s > nt else 0 for s, nt in zip(scores, neutral)]


def ssa_verdict(cell: dict) -> dict:
    """Apply the six pre-registered SSA conditions; ABSTAIN on the first failure.
    Order matters only for the reported reason; all must hold to be 'enacted'."""
    T = SSA_THRESHOLDS
    checks = {}
    lo, hi = cell["delta_ci"]
    checks["superiority"] = lo > 0.0 and cell["delta_point"] >= T["delta_point_min"]
    checks["floor"] = cell["steered_d"] > T["steered_d_min"]
    checks["orthogonality"] = all(abs(d) < T["off_target_max"] for d in cell["off_target_d"].values())
    checks["corroboration"] = cell["kappa"] >= T["kappa_floor"]
    checks["capability"] = (cell["capability_drop"] < T["capability_eps"]
                            and cell["coherence"] >= T["coherence_floor"])
    checks["non_mock"] = not cell["is_mock"]
    reason_for = {
        "superiority": "steer_not_beats_baseline", "floor": "below_floor",
        "orthogonality": "off_target_halo", "corroboration": "low_kappa",
        "capability": "capability_drop", "non_mock": "mock_subject",
    }
    for key in ("non_mock", "superiority", "floor", "orthogonality", "corroboration", "capability"):
        if not checks[key]:
            return {"status": "abstained", "reason": reason_for[key], "checks": checks}
    return {"status": "enacted", "reason": None, "checks": checks}
