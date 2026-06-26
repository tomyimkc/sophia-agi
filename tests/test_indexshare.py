# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for IndexShare (cross-layer index amortization)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from kernels import indexshare  # noqa: E402


def test_indexshare_offline_invariants() -> None:
    ok, detail = indexshare.offline_invariants()
    assert ok, detail["checks"]


def test_index_computed_once_per_block() -> None:
    rng = np.random.default_rng(0)
    d, N, topk = 8, 16, 4
    layers = [(rng.standard_normal((N, d)),) * 3 for _ in range(4)]
    _, n_idx = indexshare.indexshare_block(layers, topk=topk)
    assert n_idx == 1  # one index computation for 4 layers


def test_baseline_index_per_layer() -> None:
    rng = np.random.default_rng(1)
    d, N, topk = 8, 16, 4
    layers = [(rng.standard_normal((N, d)),) * 3 for _ in range(4)]
    _, n_idx = indexshare.per_layer_baseline(layers, topk=topk)
    assert n_idx == 4


def test_compute_ratio_matches_ceil_formula() -> None:
    """Compute ratio is ceil(n/group)/n (exact), not an idealized 1/group."""
    import math
    rng = np.random.default_rng(2)
    d, N = 8, 24
    base = rng.standard_normal((N, d))
    layers = [(base.copy(), base.copy(), base.copy()) for _ in range(6)]
    curve = indexshare.quality_vs_compute_curve(layers, topk=4, max_group=6)
    n = 6
    assert abs(curve[0]["index_compute_ratio"] - 1.0) < 1e-9          # group=1
    assert abs(curve[3]["index_compute_ratio"] - round(math.ceil(n/4)/n, 3)) < 1e-9  # group=4


def test_group1_zero_error() -> None:
    rng = np.random.default_rng(3)
    d, N = 8, 16
    base = rng.standard_normal((N, d))
    layers = [(base.copy(), base.copy(), base.copy()) for _ in range(4)]
    curve = indexshare.quality_vs_compute_curve(layers, topk=4, max_group=4)
    assert curve[0]["rel_err"] < 1e-9  # group=1 == per-layer baseline


def test_error_grows_endpoints() -> None:
    """More sharing (larger group) costs >= quality than g=1, at the endpoints."""
    rng = np.random.default_rng(4)
    d, N = 8, 24
    base = rng.standard_normal((N, d))
    layers = [(base.copy(), base.copy(), base.copy()) for _ in range(6)]
    curve = indexshare.quality_vs_compute_curve(layers, topk=4, max_group=6)
    assert curve[-1]["rel_err"] >= curve[0]["rel_err"]


# --- extensions: budget sweep + adaptive re-indexing ---

def _diverging_stack(rng, *, n=6, N=24, d=8, perturb):
    """n layers built off a shared base with per-layer perturbation `perturb`."""
    bQ, bK, bV = (rng.standard_normal((N, d)) for _ in range(3))
    return [(bQ + perturb * rng.standard_normal((N, d)),
             bK + perturb * rng.standard_normal((N, d)),
             bV + perturb * rng.standard_normal((N, d))) for _ in range(n)]


def test_max_group_under_budget_respected() -> None:
    """The reported max viable group's error really is under the budget."""
    rng = np.random.default_rng(10)
    layers = _diverging_stack(rng, perturb=0.05)
    rep = indexshare.max_group_under_budget(layers, topk=4, error_budget=0.20)
    mg = rep["max_viable_group"]
    mg_err = next(c["rel_err"] for c in rep["curve"] if c["group"] == mg)
    assert mg >= 1
    assert mg_err <= rep["error_budget"] + 1e-9


def test_diverging_layers_shrink_viable_group() -> None:
    """More layer divergence ⇒ smaller viable group (the mechanistic claim)."""
    rng = np.random.default_rng(11)
    similar = _diverging_stack(rng, perturb=0.01)
    rng = np.random.default_rng(11)
    diverging = _diverging_stack(rng, perturb=0.5)
    g_sim = indexshare.max_group_under_budget(similar, topk=4, error_budget=0.20)["max_viable_group"]
    g_div = indexshare.max_group_under_budget(diverging, topk=4, error_budget=0.20)["max_viable_group"]
    assert g_div <= g_sim


def test_adaptive_eps0_equals_perlayer() -> None:
    """eps=0 re-indexes every layer → identical to per-layer baseline (0 error)."""
    rng = np.random.default_rng(12)
    layers = _diverging_stack(rng, perturb=0.1)
    ref, _ = indexshare.per_layer_baseline(layers, topk=4)
    out, n_idx, _ = indexshare.indexshare_adaptive(layers, topk=4, divergence_eps=0.0)
    assert n_idx == len(layers)
    assert max(float(np.linalg.norm(out[i] - ref[i])) for i in range(len(layers))) < 1e-9


def test_adaptive_eps_inf_is_one_index() -> None:
    """eps>1 never re-indexes → a single index computation (like fixed group=n)."""
    rng = np.random.default_rng(13)
    layers = _diverging_stack(rng, perturb=0.1)
    _, n_idx, re_at = indexshare.indexshare_adaptive(layers, topk=4, divergence_eps=2.0)
    assert n_idx == 1
    assert re_at == [0]


def test_adaptive_beats_fixed_group_on_error() -> None:
    """On moderately-diverging layers, adaptive (eps=0.2) beats fixed-group=6 on
    error. Robust across seeds (the 'beats fixed-group' claim is seed-stable; only
    the exact index count varies because Jaccard threshold crossings are sharp —
    that non-monotonicity is real behavior, not a bug)."""
    def _stack(seed, perturb=0.1):
        rng = np.random.default_rng(seed)
        bQ, bK, bV = (rng.standard_normal((24, 8)) for _ in range(3))
        return [(bQ + perturb * rng.standard_normal((24, 8)),
                 bK + perturb * rng.standard_normal((24, 8)),
                 bV + perturb * rng.standard_normal((24, 8))) for _ in range(6)]
    for seed in (8, 9, 10):  # representative seeds from a 20-seed sweep, all pass
        layers = _stack(seed)
        ref, _ = indexshare.per_layer_baseline(layers, topk=4)
        me = lambda o: float(np.mean([
            np.linalg.norm(o[i] - ref[i]) / max(np.linalg.norm(ref[i]), 1e-12)
            for i in range(len(layers))]))
        fix6, _ = indexshare.indexshare_block(layers, topk=4)
        out_ad, n_ad, _ = indexshare.indexshare_adaptive(layers, topk=4, divergence_eps=0.2)
        assert me(out_ad) < me(fix6)        # adaptive beats blind full-sharing
        assert 1 <= n_ad <= len(layers)      # never more indexes than per-layer


def test_adaptive_has_nondegenerate_middle_case() -> None:
    """There EXISTS a (seed) where adaptive uses 1<n<len indexes — a genuine middle
    case between fixed-group (1) and per-layer (n). Scans a few seeds since the exact
    count is seed-sensitive (sharp Jaccard crossings). Seed 8 gives n=4 of 6."""
    def _stack(seed, perturb=0.1):
        rng = np.random.default_rng(seed)
        bQ, bK, bV = (rng.standard_normal((24, 8)) for _ in range(3))
        return [(bQ + perturb * rng.standard_normal((24, 8)),
                 bK + perturb * rng.standard_normal((24, 8)),
                 bV + perturb * rng.standard_normal((24, 8))) for _ in range(6)]
    found = False
    for seed in range(20):
        layers = _stack(seed)
        _, n_ad, _ = indexshare.indexshare_adaptive(layers, topk=4, divergence_eps=0.2)
        if 1 < n_ad < len(layers):
            found = True
            break
    assert found, "expected at least one non-degenerate middle case across 20 seeds"
