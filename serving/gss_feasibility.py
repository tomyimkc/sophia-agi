# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Governed Speculative Sparsity — Tier-0 feasibility meter (the cheap go/no-go).

*Why this exists.* Before spending a single GPU-hour on Governed Speculative Sparsity
(GSS — see ``docs/11-Platform/Real-Tensor-Movement-Thesis.md``), two numbers decide
whether it can possibly beat a dense decode on a bandwidth-bound machine:

  - **ρ (read-set fraction)** — for each token, the *minimal fraction of weight units*
    (MoE experts, or MLP/attention channels) needed to carry ``coverage`` of the
    output's contribution mass. Small ρ ⇒ most weights don't move the logits this
    token ⇒ they are skippable. This is the "how few bytes must I read?" lever.
  - **k (expected accepted tokens per verify pass)** — the speculative-decoding
    amortization factor for a 4-bit *self-draft*: how many tokens one expensive target
    verification produces. Large k ⇒ the verify cost is spread thin.

These two, plus the draft's byte cost, fully determine the GSS roofline cost ratio
(``§4.3`` of the thesis). If the ratio is ≥ 1, **GSS cannot win and is abandoned here
— for the price of a CPU run, no GPU spent.** That fail-closed kill switch is the
entire point of Tier 0.

*What it does.* Pure-numpy measurement over arrays the caller produced from a real
forward pass:
  - ``read_set_fraction`` — ρ from per-token, per-unit contribution magnitudes.
  - ``acceptance_rate`` — the speculative acceptance α = 1 − TV(target ‖ draft) from
    paired next-token distributions (a 4-bit self-pass vs FP16).
  - ``expected_accepted`` — k = (1 − α^(γ+1)) / (1 − α), the Leviathan et al. expected
    tokens-per-block (bounded by γ+1).
  - ``GSSFeasibilityGate.evaluate`` — combine into the honest cost ratio and a GO/NO-GO.

*The honest cost model (stricter than the thesis headline).* Per produced token, GSS
reads, in units of the dense per-token weight bytes ``B``:
    draft  = γ · ``draft_byte_frac`` · B   (γ *autoregressive* 4-bit self-passes; weights
                                            re-read each token — drafting can't batch)
    verify = ρ · B                          (ONE parallel target pass over the γ-token
                                            block; weights amortized across the batch)
    per token ≈ B · (γ·draft_byte_frac + ρ) / k
So ``cost_ratio = (γ·draft_byte_frac + ρ) / k`` and GSS can win **iff cost_ratio < 1**.
(The thesis's simplified ``(0.25+ρ)/k`` collapses the γ draft passes to one and is
optimistic; this module uses the per-block draft accounting and is the number to trust.)

*Model-agnostic by construction.* It consumes arrays, never a model, so it is
CI-testable on synthetic activations; the caller owns the real forward passes (the
same contract as :mod:`serving.lowram_eval`). It is a *feasibility meter*, not the GSS
mechanism itself (that is Tier 1, ``serving/gss.py``) — and certainly not a speedup
claim. ``canClaimAGI`` stays ``false``.

Falsifiable offline invariants (``offline_invariants()``, CI-gated):
  - a concentrated read-set yields small ρ; a uniform one yields ρ ≈ coverage;
  - ρ is monotonic non-decreasing in coverage (asking for more mass reads ≥ as many units);
  - identical draft==target gives α=1 and k=γ+1; divergence lowers both, k∈[1, γ+1];
  - a concentrated + high-acceptance regime is GO (cost_ratio<1, ceiling>1); a diffuse +
    low-acceptance regime is NO-GO — the kill switch bites;
  - the report always carries ρ, k *and* the cost ratio together, never one alone;
  - fail-closed on empty / shape-mismatch / negative-contrib / non-finite / bad coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# ρ — read-set fraction (how few weight units carry the output mass)
# ---------------------------------------------------------------------------

def read_set_fraction(contribs, *, coverage: float = 0.9):
    """Mean minimal read-set fraction ρ over tokens.

    ``contribs`` : (T, U) non-negative per-token contribution magnitudes over U weight
        units (e.g. per-expert gate·norm, or per-channel |activation·weight|). Row t,
        unit u = how much unit u moved token t's output.
    ``coverage`` : fraction of each row's total mass the read-set must capture (0,1].

    Returns ``(rho_mean, per_token_fraction)``. A row whose mass is 0 cannot be pruned
    safely → its fraction is 1.0 (read everything), fail-closed.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1]")
    c = np.asarray(contribs, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("contribs must be 2-D (T, U)")
    if c.shape[0] == 0 or c.shape[1] == 0:
        raise ValueError("contribs must be non-empty")
    if not np.isfinite(c).all():
        raise ValueError("contribs must be finite")
    if (c < 0).any():
        raise ValueError("contribs must be non-negative magnitudes")

    u = c.shape[1]
    s = np.sort(c, axis=1)[:, ::-1]          # descending per row
    csum = np.cumsum(s, axis=1)              # (T, U) increasing
    total = csum[:, -1:]                     # (T, 1)
    thresh = coverage * total                # mass each row must reach
    # minimal count to reach the threshold = (#units strictly below thresh) + 1
    need = (csum < thresh).sum(axis=1) + 1
    need = np.clip(need, 1, u)
    frac = need / u
    frac = np.where(total[:, 0] <= 0, 1.0, frac)   # zero-mass rows can't prune
    return float(frac.mean()), frac


def read_set_temporal_stability(contribs, *, coverage: float = 0.9) -> float:
    """Mean Jaccard overlap of consecutive tokens' read-sets — a *reported* diagnostic.

    High stability means the same units recur token-to-token (cache/prefetch friendly);
    it is informative but **not** part of the GO/NO-GO gate (ρ and k are). Returns 0.0
    for a single token (no consecutive pair).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    c = np.asarray(contribs, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] == 0 or c.shape[1] == 0:
        raise ValueError("contribs must be a non-empty 2-D (T, U)")
    t, u = c.shape
    if t < 2:
        return 0.0
    order = np.argsort(-c, axis=1)
    csum = np.cumsum(np.take_along_axis(c, order, axis=1), axis=1)
    total = csum[:, -1:]
    thresh = coverage * np.where(total <= 0, 1.0, total)
    need = (csum < thresh).sum(axis=1) + 1
    need = np.clip(need, 1, u)
    sets = [set(order[i, : need[i]].tolist()) for i in range(t)]
    jac = []
    for i in range(t - 1):
        a, b = sets[i], sets[i + 1]
        union = a | b
        jac.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(jac))


# ---------------------------------------------------------------------------
# k — speculative acceptance of a 4-bit self-draft
# ---------------------------------------------------------------------------

def acceptance_rate(target_probs, draft_probs):
    """Mean speculative acceptance α = 1 − TV(target ‖ draft) over positions.

    ``target_probs`` / ``draft_probs`` : (P, V) next-token distributions over the same P
    positions (rows realigned, e.g. FP16 target vs 4-bit self-draft). Under speculative
    sampling the marginal accept probability of a drafted token equals
    ``Σ_x min(p(x), q(x)) = 1 − TV(p, q)``. Returns ``(alpha_mean, per_position_alpha)``.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    p = np.asarray(target_probs, dtype=np.float64)
    q = np.asarray(draft_probs, dtype=np.float64)
    if p.ndim != 2 or q.ndim != 2:
        raise ValueError("probs must be 2-D (P, V)")
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch target{p.shape} vs draft{q.shape}")
    if p.shape[0] == 0:
        raise ValueError("no positions")
    if not (np.isfinite(p).all() and np.isfinite(q).all()):
        raise ValueError("probs must be finite")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("probs must be non-negative")
    ps = p.sum(axis=1, keepdims=True)
    qs = q.sum(axis=1, keepdims=True)
    if (ps <= 0).any() or (qs <= 0).any():
        raise ValueError("each distribution must have positive mass")
    p = p / ps
    q = q / qs
    alpha_rows = np.minimum(p, q).sum(axis=1)   # 1 - TV per position
    return float(alpha_rows.mean()), alpha_rows


def expected_accepted(alpha: float, gamma: int) -> float:
    """Expected tokens produced per verify pass: k = (1 − α^(γ+1)) / (1 − α).

    The Leviathan et al. block expectation, including the target's bonus token; bounded
    above by γ+1 (reached as α→1). ``gamma`` = speculative block size (draft length).
    """
    if gamma < 1:
        raise ValueError("gamma must be >= 1")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")
    if alpha >= 1.0:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


# ---------------------------------------------------------------------------
# The feasibility gate — the honest cost ratio and the GO/NO-GO verdict
# ---------------------------------------------------------------------------

@dataclass
class GSSFeasibilityReport:
    """ρ, k, and the cost ratio that decides them — always reported together."""

    go: bool
    rho: float                 # mean read-set fraction at the coverage threshold
    alpha: float               # mean speculative acceptance of the self-draft
    k: float                   # expected accepted tokens per verify pass
    cost_ratio: float          # (gamma*draft_byte_frac + rho) / k ; < 1 ⇒ can win
    speedup_ceiling: float     # 1 / cost_ratio (roofline bandwidth-reduction ceiling)
    gamma: int
    draft_byte_frac: float
    coverage: float
    temporal_stability: float  # reported diagnostic, not gated
    n_tokens: int
    n_positions: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "go": self.go,
            "rho": round(self.rho, 6),
            "alpha": round(self.alpha, 6),
            "k": round(self.k, 6),
            "cost_ratio": round(self.cost_ratio, 6),
            "speedup_ceiling": round(self.speedup_ceiling, 4),
            "gamma": self.gamma,
            "draft_byte_frac": self.draft_byte_frac,
            "coverage": self.coverage,
            "temporal_stability": round(self.temporal_stability, 4),
            "n_tokens": self.n_tokens,
            "n_positions": self.n_positions,
            "reasons": self.reasons,
        }


@dataclass
class GSSFeasibilityGate:
    """Tier-0 GO/NO-GO for Governed Speculative Sparsity.

    ``gamma``           : speculative block size (draft length per verify pass).
    ``coverage``        : output mass the per-token read-set must capture.
    ``draft_byte_frac`` : draft read bytes per weight unit ÷ target's (4-bit/16-bit = 0.25).
    ``max_cost_ratio``  : GO iff cost_ratio < this (default 1.0 = must beat dense). Set
                          < 1.0 to demand a margin before greenlighting GPU work.
    """

    gamma: int = 4
    coverage: float = 0.9
    draft_byte_frac: float = 0.25
    max_cost_ratio: float = 1.0

    def evaluate(self, contribs, target_probs, draft_probs) -> GSSFeasibilityReport:
        """Measure ρ, α, k from real arrays and apply the cost-model gate (fail-closed)."""
        if not _HAVE_NUMPY:
            raise RuntimeError("numpy required")
        reasons: list[str] = []

        def _reject(reason: str) -> GSSFeasibilityReport:
            return GSSFeasibilityReport(
                go=False, rho=1.0, alpha=0.0, k=1.0, cost_ratio=float("inf"),
                speedup_ceiling=0.0, gamma=self.gamma, draft_byte_frac=self.draft_byte_frac,
                coverage=self.coverage, temporal_stability=0.0, n_tokens=0, n_positions=0,
                reasons=[reason],
            )

        if self.gamma < 1:
            return _reject("gamma must be >= 1")
        if not (0.0 < self.coverage <= 1.0):
            return _reject("coverage must be in (0, 1]")
        if not (0.0 < self.draft_byte_frac <= 1.0):
            return _reject("draft_byte_frac must be in (0, 1]")
        try:
            rho, _ = read_set_fraction(contribs, coverage=self.coverage)
            stab = read_set_temporal_stability(contribs, coverage=self.coverage)
            alpha, _ = acceptance_rate(target_probs, draft_probs)
            k = expected_accepted(alpha, self.gamma)
        except (ValueError, RuntimeError) as e:
            return _reject(f"input rejected: {e}")

        cost_ratio = (self.gamma * self.draft_byte_frac + rho) / k
        ceiling = (1.0 / cost_ratio) if cost_ratio > 0 else float("inf")
        go = cost_ratio < self.max_cost_ratio
        if not go:
            reasons.append(
                f"cost_ratio {cost_ratio:.4f} >= {self.max_cost_ratio}: "
                f"GSS cannot beat dense at ρ={rho:.3f}, k={k:.3f}, γ={self.gamma} — abandon (no GPU)"
            )

        c = np.asarray(contribs)
        p = np.asarray(target_probs)
        return GSSFeasibilityReport(
            go=go, rho=rho, alpha=alpha, k=k, cost_ratio=cost_ratio, speedup_ceiling=ceiling,
            gamma=self.gamma, draft_byte_frac=self.draft_byte_frac, coverage=self.coverage,
            temporal_stability=stab, n_tokens=int(c.shape[0]), n_positions=int(p.shape[0]),
            reasons=reasons,
        )


# ---------------------------------------------------------------------------
# Confidence intervals — within-run (bootstrap over positions) and across-run
# ---------------------------------------------------------------------------

def bootstrap_ci(values, *, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap ``(lo, hi)`` for the mean of ``values``.

    A cheap honesty upgrade: turns a point estimate into an interval over the sample it was
    computed on. Returns ``(lo, hi)`` at the central ``ci`` mass (default 95%).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("values must be non-empty")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1)")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return lo, hi


def feasibility_with_ci(contribs, target_probs, draft_probs, *, gamma: int = 4,
                        coverage: float = 0.9, draft_byte_frac: float = 0.25,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """Point estimates **and** within-run bootstrap 95% CIs for ρ, α, k, and cost_ratio.

    Resamples positions jointly (the ρ read-fraction and the α acceptance share the index),
    recomputing the whole cost model per bootstrap draw, so the cost_ratio CI is honest about
    the (ρ, α) covariance. Caveat: a *within-run* CI captures sampling over positions, not
    run-to-run variance — a registered result still needs ≥3 runs (use ``aggregate_runs``).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    _, rho_per = read_set_fraction(contribs, coverage=coverage)
    _, alpha_per = acceptance_rate(target_probs, draft_probs)
    n = min(rho_per.size, alpha_per.size)
    rho_per, alpha_per = rho_per[:n], alpha_per[:n]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    rho_b = rho_per[idx].mean(axis=1)
    alpha_b = np.clip(alpha_per[idx].mean(axis=1), 0.0, 1.0)
    k_b = np.where(alpha_b >= 1.0, gamma + 1.0,
                   (1.0 - alpha_b ** (gamma + 1)) / (1.0 - np.minimum(alpha_b, 1 - 1e-12)))
    cost_b = (gamma * draft_byte_frac + rho_b) / k_b

    def _q(a):
        return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]
    rho, k = float(rho_per.mean()), expected_accepted(float(alpha_per.mean()), gamma)
    alpha = float(alpha_per.mean())
    cost = (gamma * draft_byte_frac + rho) / k
    return {
        "n": int(n), "gamma": gamma, "coverage": coverage, "draft_byte_frac": draft_byte_frac,
        "rho": rho, "rho_ci95": _q(rho_b),
        "alpha": alpha, "alpha_ci95": _q(alpha_b),
        "k": k, "k_ci95": _q(k_b),
        "cost_ratio": cost, "cost_ratio_ci95": _q(cost_b),
        "go": bool(cost < 1.0), "go_ci_excludes_1": bool(_q(cost_b)[1] < 1.0),
    }


def aggregate_runs(reports, *, draft_byte_frac: float = 0.25) -> dict:
    """Across-run aggregate for a ≥3-run campaign — the *registered* confidence statement.

    ``reports`` is a list of per-run dicts (each a ``GSSFeasibilityReport.as_dict()`` or
    ``feasibility_with_ci`` result). Returns mean and a normal 95% CI (mean ± 1.96·SE) for
    ρ, α, k, cost_ratio across runs, and whether the cost_ratio CI upper bound stays < 1
    (the no-overclaim "GO excludes 1" bar). Run-to-run variance is what the within-run
    bootstrap cannot see, so this is the statement that licenses a registered result.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    rows = [r for r in reports if r]
    if len(rows) < 2:
        raise ValueError("need >=2 runs to estimate run-to-run variance")
    out: dict = {"n_runs": len(rows)}
    for key in ("rho", "alpha", "k", "cost_ratio"):
        vals = np.array([float(r[key]) for r in rows], dtype=np.float64)
        mean = float(vals.mean())
        se = float(vals.std(ddof=1) / np.sqrt(vals.size))
        out[key] = mean
        out[f"{key}_ci95"] = [mean - 1.96 * se, mean + 1.96 * se]
    out["go"] = bool(out["cost_ratio"] < 1.0)
    out["go_ci_excludes_1"] = bool(out["cost_ratio_ci95"][1] < 1.0)
    return out


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    T, U, V = 48, 64, 100

    # --- ρ behaviour ---------------------------------------------------------
    # Concentrated: each token's mass sits in a few units -> small ρ.
    conc = np.full((T, U), 1e-3)
    for t in range(T):
        hot = rng.choice(U, size=4, replace=False)
        conc[t, hot] += rng.uniform(5, 10, size=4)
    rho_conc, _ = read_set_fraction(conc, coverage=0.9)
    checks["concentrated_low_rho"] = rho_conc < 0.25
    detail["rho_concentrated"] = round(rho_conc, 4)

    # Uniform: need ~coverage of the units -> ρ ≈ coverage.
    uni = np.ones((T, U))
    rho_uni, _ = read_set_fraction(uni, coverage=0.9)
    checks["uniform_rho_near_coverage"] = abs(rho_uni - 0.9) <= 1.0 / U + 1e-9
    detail["rho_uniform"] = round(rho_uni, 4)

    # Monotone in coverage.
    r50, _ = read_set_fraction(conc, coverage=0.5)
    r99, _ = read_set_fraction(conc, coverage=0.99)
    checks["rho_monotone_in_coverage"] = r50 <= rho_conc <= r99 + 1e-12
    detail["rho_sweep"] = {"c50": round(r50, 4), "c90": round(rho_conc, 4), "c99": round(r99, 4)}

    # --- α / k behaviour -----------------------------------------------------
    logits = rng.standard_normal((T, V)) * 3.0
    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)
    p = softmax(logits)

    # Identical self-draft: α=1, k=γ+1.
    a_id, _ = acceptance_rate(p, p.copy())
    k_id = expected_accepted(a_id, 4)
    checks["identical_alpha_one"] = abs(a_id - 1.0) < 1e-9 and abs(k_id - 5.0) < 1e-9
    detail["identical"] = {"alpha": round(a_id, 4), "k": round(k_id, 4)}

    # A faithful 4-bit-like draft (small logit noise): high α, k strictly < γ+1 but large.
    q_good = softmax(logits + rng.standard_normal((T, V)) * 0.4)
    a_good, _ = acceptance_rate(p, q_good)
    k_good = expected_accepted(a_good, 4)
    checks["good_draft_high_alpha"] = a_good > 0.6
    checks["k_bounded"] = 1.0 <= k_good <= 5.0
    detail["good_draft"] = {"alpha": round(a_good, 4), "k": round(k_good, 4)}

    # A poor draft (heavy noise): low α, k -> ~1.
    q_bad = softmax(logits + rng.standard_normal((T, V)) * 6.0)
    a_bad, _ = acceptance_rate(p, q_bad)
    k_bad = expected_accepted(a_bad, 4)
    checks["bad_draft_low_alpha"] = a_bad < a_good
    detail["bad_draft"] = {"alpha": round(a_bad, 4), "k": round(k_bad, 4)}

    # --- the gate: GO and NO-GO regimes -------------------------------------
    gate = GSSFeasibilityGate(gamma=4, coverage=0.9, draft_byte_frac=0.25)

    go_rep = gate.evaluate(conc, p, q_good)         # concentrated + faithful -> GO
    checks["go_regime_go"] = go_rep.go and go_rep.cost_ratio < 1.0 and go_rep.speedup_ceiling > 1.0
    detail["go_report"] = go_rep.as_dict()

    nogo_rep = gate.evaluate(uni, p, q_bad)         # diffuse + poor -> NO-GO (kill switch)
    checks["nogo_regime_stops"] = (not nogo_rep.go) and nogo_rep.cost_ratio >= 1.0
    detail["nogo_report"] = nogo_rep.as_dict()

    # Report always carries ρ, k and cost_ratio together (never a partial verdict).
    d = go_rep.as_dict()
    checks["report_is_complete"] = all(d[key] is not None for key in ("rho", "k", "cost_ratio", "go"))

    # Temporal stability is in [0,1]; a fixed hot-set stream is highly stable.
    fixed = np.full((T, U), 1e-3)
    fixed[:, :4] += 5.0
    stab = read_set_temporal_stability(fixed, coverage=0.9)
    checks["stability_in_range"] = 0.0 <= stab <= 1.0
    checks["fixed_set_is_stable"] = stab > 0.9
    detail["stability"] = round(stab, 4)

    # --- fail-closed ---------------------------------------------------------
    bad_cases = {
        "empty_contribs": lambda: gate.evaluate(np.zeros((0, U)), p, q_good),
        "shape_mismatch_probs": lambda: gate.evaluate(conc, p, q_good[:, : V - 1]),
        "negative_contribs": lambda: gate.evaluate(-conc, p, q_good),
        "nonfinite_probs": lambda: gate.evaluate(conc, p * np.inf, q_good),
    }
    for name, fn in bad_cases.items():
        rep = fn()
        checks[f"failclosed_{name}"] = (not rep.go) and rep.cost_ratio == float("inf")

    # Bad coverage / gamma rejected at gate construction-time semantics.
    checks["failclosed_bad_coverage"] = not GSSFeasibilityGate(coverage=1.5).evaluate(conc, p, q_good).go
    checks["failclosed_bad_gamma"] = not GSSFeasibilityGate(gamma=0).evaluate(conc, p, q_good).go

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("GSS feasibility offline invariants:", "PASS" if ok else "FAIL")
    for k_, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k_}")
    print(f"  GO report:   {detail.get('go_report')}")
    print(f"  NO-GO report:{detail.get('nogo_report')}")
    raise SystemExit(0 if ok else 1)
