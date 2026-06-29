# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Representation-level shortcut probe — detect reward-hacking by its signature.

The static scan and the isolated grader (``code_integrity`` / ``code_exec``) stop
*known* cheat shapes. *When Reward Hacking Rebounds* (arXiv:2604.01476) shows the
deeper signal: hacking has a characteristic **representation-level direction** in a
model's activations that tracks it across an RL run — independent of any syntactic
rule, so it can catch a *novel* exploit the scanner has no pattern for. The paper
folds the probe's score into the GRPO **advantage** ("Advantage Modification") so
hacking rollouts are suppressed at the training signal, anticipating the *rebound*
(blocking the surface without penalizing the intent just teaches a new trick).

This module is the **offline-testable scaffold** for that defence: fit a linear
"shortcut direction" from per-rollout feature vectors labelled hack/clean, score
new rollouts, and modify advantages to penalize shortcut-like rollouts. It operates
on **injected** feature vectors so it is fully testable with no model — the LIVE
piece (capturing real hidden-state features during GRPO) is the open work; this is
deliberately the same "wire + offline-prove, then GPU" pattern the repo uses
elsewhere. Pure stdlib, deterministic. Candidate, not validated; ``canClaimAGI``
unaffected.
"""

from __future__ import annotations

import math
from typing import Sequence

Vec = Sequence[float]


def _mean(rows: list[Vec]) -> list[float]:
    n = len(rows)
    d = len(rows[0])
    out = [0.0] * d
    for r in rows:
        for i, v in enumerate(r):
            out[i] += v
    return [v / n for v in out]


def _dot(a: Vec, b: Vec) -> float:
    return sum(x * y for x, y in zip(a, b))


def fit(features: list[Vec], labels: list[bool]) -> dict:
    """Fit a linear shortcut direction (nearest-centroid / diagonal-LDA).

    ``direction = mean(hack) - mean(clean)`` (unit-normalized); the decision
    threshold ``bias`` is the midpoint projection. Higher ``score`` = more
    shortcut-like. Robust, parameter-free, and monotone — deliberately simple so
    the signal, not the classifier, is what is under test.
    """
    if len(features) != len(labels) or not features:
        raise ValueError("features and labels must be non-empty and aligned")
    hack = [f for f, y in zip(features, labels) if y]
    clean = [f for f, y in zip(features, labels) if not y]
    if not hack or not clean:
        raise ValueError("need at least one hack and one clean example")
    mu_h, mu_c = _mean(hack), _mean(clean)
    direction = [h - c for h, c in zip(mu_h, mu_c)]
    norm = math.sqrt(sum(x * x for x in direction)) or 1.0
    direction = [x / norm for x in direction]
    bias = (_dot(direction, mu_h) + _dot(direction, mu_c)) / 2.0
    return {"direction": direction, "bias": bias,
            "proj_clean": _dot(direction, mu_c), "proj_hack": _dot(direction, mu_h)}


def score(model: dict, feature: Vec) -> float:
    """Signed shortcut score: > 0 means shortcut-like (toward the hack centroid)."""
    return _dot(model["direction"], feature) - model["bias"]


def predict_hack(model: dict, feature: Vec) -> bool:
    return score(model, feature) > 0.0


def auc(model: dict, features: list[Vec], labels: list[bool]) -> float:
    """Rank AUC of the shortcut score vs the hack label (1.0 = perfect separation)."""
    scored = [(score(model, f), y) for f, y in zip(features, labels)]
    pos = [s for s, y in scored if y]
    neg = [s for s, y in scored if not y]
    if not pos or not neg:
        return 0.5
    wins = sum((sp > sn) + 0.5 * (sp == sn) for sp in pos for sn in neg)
    return wins / (len(pos) * len(neg))


def advantage_modification(advantages: list[float], features: list[Vec], model: dict,
                           *, beta: float = 1.0) -> list[float]:
    """GRPO advantage modification (arXiv:2604.01476): subtract a penalty
    proportional to each rollout's (squashed, non-negative) shortcut score, so
    shortcut-like rollouts get less advantage and are suppressed at the training
    signal rather than only filtered at eval. ``beta`` scales the penalty; a clean
    rollout (score <= 0) is left unchanged.
    """
    out = []
    for a, f in zip(advantages, features):
        s = score(model, f)
        penalty = beta * (1.0 / (1.0 + math.exp(-s))) if s > 0 else 0.0  # sigmoid on positive only
        out.append(a - penalty)
    return out


def _synthetic(seed: int = 0, d: int = 8, n: int = 120, shift: float = 2.0):
    """Deterministic two-cluster features: clean ~ N(0,1)^d, hack shifted by +shift
    along the first 3 dims. Labels: half hack, half clean."""
    import random

    rng = random.Random(seed)
    feats, labs = [], []
    for i in range(n):
        hack = i % 2 == 0
        v = [rng.gauss(0.0, 1.0) for _ in range(d)]
        if hack:
            for j in range(3):
                v[j] += shift
        feats.append(v)
        labs.append(hack)
    return feats, labs


def offline_invariants() -> "tuple[bool, dict]":
    """Prove the scaffold works with no model: on synthetic linearly-separable
    rollout features it (1) fits a direction that separates hack from clean on a
    HELD-OUT split (AUC high) and (2) its advantage modification lowers hacking
    rollouts' advantage strictly more than clean rollouts'. The LIVE caveat — real
    activation features — is named, not silently assumed."""
    feats, labs = _synthetic(seed=0)
    tr_f, tr_l = feats[:80], labs[:80]
    te_f, te_l = feats[80:], labs[80:]
    model = fit(tr_f, tr_l)
    a = auc(model, te_f, te_l)

    base = [1.0] * len(te_f)
    mod = advantage_modification(base, te_f, model, beta=1.0)
    drop_hack = [b - m for b, m, y in zip(base, mod, te_l) if y]
    drop_clean = [b - m for b, m, y in zip(base, mod, te_l) if not y]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    checks = {
        "separatesHeldOut": a >= 0.90,
        "penalizesHackMore": mean(drop_hack) > mean(drop_clean) + 0.2,
        "cleanMostlyUnpenalized": mean(drop_clean) < 0.2,
        "deterministic": fit(tr_f, tr_l)["direction"] == model["direction"],
    }
    detail = {
        "auc": round(a, 3), "meanDropHack": round(mean(drop_hack), 3),
        "meanDropClean": round(mean(drop_clean), 3), "checks": checks,
        "liveCaveat": "synthetic features; live use needs real GRPO activation capture (open)",
    }
    return all(checks.values()), detail
