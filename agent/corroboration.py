"""Corroboration-aware confidence — independent agreement should RAISE belief.

The OKF belief graph propagates confidence as **min-over-chain** (``okf/graph.py``),
which correctly stops a high-confidence claim from *laundering* through a weak
ancestor. But min-over-chain answers a different question than *corroboration*: it
cannot reward the fact that several **independent** sources agree. This module adds
that missing axis.

Design (the two failure modes the review named, both handled):
  - **Corroboration:** combine independent supports with a Bayesian **log-odds
    pool** — N independent agreeing sources push the posterior up; a dissenting
    source pushes it down. (Log-odds is used over raw Dempster–Shafer because DS
    misbehaves under high conflict — Zadeh's paradox; log-odds does not.)
  - **No double-counting:** dependent sources (same ``independence_group``) are
    collapsed to a single opinion (averaged) BEFORE pooling, so copying a source
    cannot inflate confidence.

Falsifiable (see tests + tools/run_corroboration.py). The GATED, robust win is
structural: confidence is monotone in the number of INDEPENDENT agreeing sources
(a mean-of-opinions baseline is flat, min is flat/decreasing — only this combiner
reflects independent support), idempotent under duplicates, and lowered by dissent.
On a labelled benchmark its confidence-ranked decisions show favourable selective
risk vs single/mean baselines, but that delta is REPORTED, not gated — decision
accuracy ties a mean baseline and the selective-risk margin is noisy at this N (the
same lesson as ECE; a single source is trivially calibrated). min-over-chain is
shown only for contrast (a laundering guard, not a classifier). Deterministic, no model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Evidence:
    """One source's support for a claim: ``confidence`` = P(claim true | source) in
    [0,1] (>0.5 supports, <0.5 opposes). Sources sharing an ``independence_group``
    are dependent and are not counted twice."""

    source_id: str
    confidence: float
    independence_group: str = field(default="")

    def __post_init__(self):
        c = self.confidence
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not (0.0 <= c <= 1.0):
            raise ValueError(f"confidence must be a finite probability in [0,1], got {c!r}")
        if not self.independence_group:
            object.__setattr__(self, "independence_group", self.source_id)


def _clamp(p: float, eps: float = 1e-6) -> float:
    return min(1.0 - eps, max(eps, float(p)))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _group_opinions(evidences: list) -> list:
    """Collapse each independence group to ONE opinion (mean confidence) so that
    dependent/duplicated sources cannot double-count."""
    groups: dict = {}
    for e in evidences:
        groups.setdefault(e.independence_group, []).append(float(e.confidence))
    return [sum(cs) / len(cs) for cs in groups.values()]


def corroborated_confidence(evidences: list, *, prior: float = 0.5, method: str = "logodds") -> float:
    """Combine evidence into a single P(claim true). ``logodds`` (default) is a
    Bayesian independent-opinion pool that handles agreement AND dissent;
    ``noisy_or`` is a support-only (Dempster–Shafer, no-conflict) combiner."""
    if method not in ("logodds", "noisy_or"):
        raise ValueError(f"unknown method {method!r}; use 'logodds' or 'noisy_or'")
    opinions = _group_opinions(evidences)
    if not opinions:
        return float(prior)
    if method == "noisy_or":
        prod = 1.0
        for c in opinions:
            prod *= (1.0 - _clamp(c))
        return round(1.0 - prod, 6)
    # log-odds pool: posterior_logit = prior_logit + Σ (logit(c_i) − prior_logit)
    pl = _logit(prior)
    s = pl + sum(_logit(c) - pl for c in opinions)
    return round(_sigmoid(s), 6)


# --------------------------------------------------------------------------- #
# Falsifiable benchmark — corroboration beats min-over-chain / single source.
# --------------------------------------------------------------------------- #


def _indep(*confs) -> list:
    return [Evidence(f"s{i}", c, independence_group=f"g{i}") for i, c in enumerate(confs)]


def build_benchmark(seed: int = 0, n: int = 400) -> list:
    """Labelled claims, each with K INDEPENDENT noisy voters (each correct w.p. q).
    A voter's ``confidence`` = q if it votes the claim true, else 1−q. Returns
    ``[(truth, [Evidence])]``."""
    import random

    rng = random.Random(seed * 7919 + 1)
    rows: list = []
    for _ in range(n):
        truth = rng.random() < 0.5
        k = rng.choice([1, 2, 3, 4, 5])
        q = rng.choice([0.6, 0.7, 0.8])
        evs = []
        for g in range(k):
            votes_true = truth if (rng.random() < q) else (not truth)
            evs.append(Evidence(f"s{g}", q if votes_true else (1 - q), independence_group=f"g{g}"))
        rows.append((truth, evs))
    return rows


def _method_pred(rows: list, combine) -> tuple:
    confs, correct = [], []
    for truth, evs in rows:
        p = combine(evs)
        pred = p >= 0.5
        confs.append(p if pred else 1 - p)        # confidence in the predicted label
        correct.append(pred == truth)
    return confs, correct


def run_demo(seed: int = 0) -> dict:
    """Structural invariants + a calibration comparison vs single-source and the
    min-over-chain rule (the wrong tool for corroboration)."""
    from agent import calibration as cal

    rows = build_benchmark(seed)
    methods = {
        "corroborated": lambda evs: corroborated_confidence(evs),
        "single": lambda evs: float(evs[0].confidence),                              # fair baseline
        "mean": lambda evs: sum(float(e.confidence) for e in evs) / len(evs),        # fair baseline
        "min": lambda evs: min(float(e.confidence) for e in evs),                    # contrast only (laundering guard, not a classifier)
    }
    ece = {}
    risk = {}
    for name, fn in methods.items():
        c, ok = _method_pred(rows, fn)
        ece[name] = cal.expected_calibration_error(c, ok)
        risk[name] = cal.selective_risk(c, ok, 0.5)

    c1 = corroborated_confidence(_indep(0.7))
    c2 = corroborated_confidence(_indep(0.7, 0.7))
    c3 = corroborated_confidence(_indep(0.7, 0.7, 0.7))
    dup = corroborated_confidence([Evidence("a", 0.7, "g0"), Evidence("b", 0.7, "g0"), Evidence("c", 0.7, "g0")])
    support2 = corroborated_confidence(_indep(0.7, 0.7))
    dissent = corroborated_confidence([Evidence("a", 0.7, "g0"), Evidence("b", 0.7, "g1"), Evidence("c", 0.2, "g2")])
    mean3 = (0.7 + 0.7 + 0.7) / 3       # 0.7 — flat
    min3 = 0.7                          # 0.7 — flat

    # Gate ONLY on the robust, defining property: corroboration's confidence reflects
    # the count of INDEPENDENT agreeing sources, which the baselines do not (mean is
    # flat, min is flat/decreasing). The benchmark's selective-risk/ECE deltas are
    # REPORTED, not gated: decision accuracy ties a mean-of-opinions baseline and the
    # selective-risk delta, while favourable on average, is too noisy at this N to
    # gate (same lesson as ECE). `min` is shown only for contrast (a laundering
    # guard, not a classifier).
    invariants = {
        "monotone_in_independent_sources": c1 < c2 < c3,
        "rewards_independent_agreement_unlike_baselines": c3 > mean3 and c3 > min3,
        "duplicates_do_not_inflate": abs(dup - c1) < 1e-6,
        "dissent_lowers_confidence": dissent < support2,
    }
    return {
        "ece": ece, "selectiveRisk": risk,
        "curve": {"1src": c1, "2src": c2, "3src": c3, "dup3same": dup, "dissent": dissent},
        "invariants": invariants, "ok": all(invariants.values()),
    }
