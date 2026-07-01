# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verified world-model training scaffold — the Phase C1 research seam.

The deliberation-roofline result says the *verifier* sets the ceiling. A world
model that predicts action outcomes, *verified against real traces*, can raise
that ceiling — but ONLY if it generalizes beyond its training traces. The
existing ``agent/predictive_world_model.py`` is an honest lookup table over a
handful of traces (the docs say so). This module is the disciplined loop that
turns "a lookup table" into "a verified, held-out-validated, shift-checked
learnable predictor" — without yet claiming the generalization problem is solved.

The structure mirrors AlphaGo's recipe applied to agentic tasks, gated by Sophia's
canary discipline (promote only on a proven held-out gain, hold on a tie, roll
back on a regression — same rule as ``selfextend/evolve.py``):

    ingest harness traces (state, action) -> outcome
      -> train an injected predictor on a TRAIN split (default: feature logistic)
      -> validate on a HELD-OUT split (did it learn, or memorize?)
      -> check generalization on a SHIFT split (different state/action distribution)
      -> promote ONLY when held-out accuracy clears a bar AND shift-degradation is bounded
      -> else hold the lookup-table baseline (fail-closed: never ship a model that
         doesn't generalize — it would silently mislead the planner)

The TRAIN STEP (the predictor) is INJECTED: the default is a dependency-free
feature-logistic model so the scaffold runs in CI; a torch-based outcome predictor
is the live seam. The scaffold trains nothing neural itself and claims no
generalization result — it provides the *measurement structure* around which a
real model is later fit, exactly the project's "interface + toy reference" idiom.

Honest scope: generalization under genuine distribution shift is the deepest open
research risk for Level-3 and is not solved here. The scaffold's job is to make
that risk VISIBLE and MEASURED — a predictor that aces held-out but collapses on
shift is flagged `shiftDegenerate`, not promoted, so the planner is never misled.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

# A predictor maps (state, action) -> P(success) in [0,1]. Injected so CI uses a
# stdlib model and the live seam uses a trained torch predictor. Trained by the
# factory's .fit(pairs) -> self contract below.
OutcomePair = tuple[str, str, int]  # (state, action, success 0/1)


@dataclass
class TrainValShift:
    """Three non-overlapping splits. ``shift`` is drawn from a DIFFERENT
    state/action distribution than train+val — the generalization probe."""

    train: list[OutcomePair]
    val: list[OutcomePair]
    shift: list[OutcomePair]


@dataclass(frozen=True)
class WorldModelReport:
    schema: str
    candidate_only: bool
    level3_evidence: bool
    predictor_kind: str
    train_size: int
    val_size: int
    shift_size: int
    val_accuracy: float
    shift_accuracy: float
    shift_degradation: float  # val - shift (positive = worse under shift)
    promoted: bool
    verdict: str  # promote | hold-shift-degenerate | hold-below-bar | hold-at-majority-baseline
    reason: str
    baseline_val_accuracy: float = 0.0  # the lookup-table baseline it had to beat

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidateOnly": self.candidate_only,
            "level3Evidence": self.level3_evidence,
            "predictorKind": self.predictor_kind,
            "trainSize": self.train_size,
            "valSize": self.val_size,
            "shiftSize": self.shift_size,
            "valAccuracy": round(self.val_accuracy, 4),
            "shiftAccuracy": round(self.shift_accuracy, 4),
            "shiftDegradation": round(self.shift_degradation, 4),
            "promoted": self.promoted,
            "verdict": self.verdict,
            "reason": self.reason,
            "baselineValAccuracy": round(self.baseline_val_accuracy, 4),
            "interpretation": _interpret(self),
        }


def _interpret(r: "WorldModelReport") -> str:
    if r.verdict == "promote":
        return (
            f"Promoted: the predictor cleared the held-out bar ({r.val_accuracy:.2f}) AND held "
            f"under shift (degradation {r.shift_degradation:.2f}). It generalized beyond its "
            f"training traces — the lookup table could not. (Candidate; not Level-3 evidence.)"
        )
    if r.verdict == "hold-shift-degenerate":
        return (
            f"Held: the predictor aced held-out ({r.val_accuracy:.2f}) but collapsed under shift "
            f"({r.shift_accuracy:.2f}, degradation {r.shift_degradation:.2f}) — it memorized the "
            f"train distribution, it did NOT generalize. The lookup-table baseline is kept so the "
            f"planner is not misled by an overfit model. This is the core Level-3 research risk."
        )
    return (
        f"Held: the predictor did not clear the held-out bar ({r.val_accuracy:.2f} < required). "
        f"Either the signal is too weak or the predictor too simple; baseline kept."
    )


class FeatureLogisticPredictor:
    """Default injected predictor: dependency-free logistic regression over a
    bag-of-(state,action)-token features. No torch, deterministic, runs in CI.

    Not a neural model — it is the toy reference that proves the scaffold wires
    end-to-end. The live seam replaces this with a trained torch outcome model."""

    def __init__(self, *, lr: float = 0.1, epochs: int = 200, l2: float = 0.01) -> None:
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: dict[str, float] = {}
        self.bias: float = 0.0

    @staticmethod
    def _features(state: str, action: str) -> list[str]:
        toks = (str(state) + " " + str(action)).lower().split()
        return [f"s:{t}" for t in toks] + [f"a:{t}" for t in toks] + [f"sa:{state.lower()}|{action.lower()}"]

    def _score(self, state: str, action: str) -> float:
        s = self.bias
        for f in self._features(state, action):
            s += self.weights.get(f, 0.0)
        return s

    def fit(self, pairs: list[OutcomePair]) -> "FeatureLogisticPredictor":
        for _ in range(self.epochs):
            for state, action, label in pairs:
                p = 1.0 / (1.0 + math.exp(-self._score(state, action)))
                err = p - float(label)
                for f in self._features(state, action):
                    self.weights[f] = self.weights.get(f, 0.0) - self.lr * (err + self.l2 * self.weights.get(f, 0.0))
                self.bias -= self.lr * err
        return self

    def predict(self, state: str, action: str) -> float:
        return 1.0 / (1.0 + math.exp(-self._score(state, action)))


def accuracy(predictor, pairs: list[OutcomePair]) -> float:
    """Fraction of pairs whose argmax (p vs 1-p) matches the label."""
    if not pairs:
        return 0.0
    correct = sum(int((predictor.predict(s, a) >= 0.5) == bool(label)) for s, a, label in pairs)
    return round(correct / len(pairs), 4)


def majority_class_accuracy(pairs: list[OutcomePair]) -> float:
    """Accuracy of the trivial always-predict-the-modal-label baseline.

    A learned predictor that merely matches this baseline has learned NOTHING — it
    reproduces the class prior. On a pass-skewed corpus (e.g. strong-model traces where
    ~95% of steps pass) this baseline is already ~0.95, so a ``val_bar`` of 0.65 is
    cleared by the trivial solution. The promote gate must require the predictor to
    STRICTLY BEAT this baseline, otherwise it promotes a majority-class mimic.
    """
    if not pairs:
        return 0.0
    labels = [int(bool(p[2])) for p in pairs]
    mode = 1 if sum(labels) >= len(labels) / 2 else 0
    return round(sum(1 for l in labels if l == mode) / len(labels), 4)


def make_splits(
    traces: list[OutcomePair],
    *,
    shift_states: set[str] | None = None,
    val_frac: float = 0.3,
    shift_frac: float = 0.2,
    seed: int = 0,
) -> TrainValShift:
    """Split traces into train / val / shift. ``shift`` is carved out as the
    fraction of traces whose state is in ``shift_states`` (a DIFFERENT distribution
    by construction); if none given, the last ``shift_frac`` of unique states are
    treated as the shift distribution. Deterministic given ``seed``."""
    if not traces:
        return TrainValShift([], [], [])
    states = sorted({s for s, _, _ in traces})
    if shift_states is None:
        k = max(1, int(len(states) * shift_frac))
        shift_states = set(states[-k:]) if k < len(states) else {states[-1]}
    shift = [t for t in traces if t[0] in shift_states]
    in_dist = [t for t in traces if t[0] not in shift_states]
    rng = random.Random(seed)
    rng.shuffle(in_dist)
    k = max(1, int(len(in_dist) * val_frac)) if in_dist else 0
    val = in_dist[:k]
    train = in_dist[k:]
    return TrainValShift(train=train, val=val, shift=shift)


def train_verified_world_model(
    traces: list[OutcomePair],
    *,
    predictor_factory: Callable[[], Any] | None = None,
    val_bar: float = 0.7,
    max_shift_degradation: float = 0.15,
    baseline_predictor: Any | None = None,
    splits: TrainValShift | None = None,
    seed: int = 0,
) -> WorldModelReport:
    """Train, validate on held-out, check generalization on shift, canary-gate.

    The canary mirrors ``selfextend/evolve.canary``: promote only when held-out
    accuracy clears ``val_bar`` AND shift-degradation is within
    ``max_shift_degradation``. Tie or regression => hold the baseline (fail-closed:
    an overfit world model is worse than the lookup table because it would
    confidently mislead the planner on states it never truly learned).
    """
    splits = splits or make_splits(traces, seed=seed)
    factory = predictor_factory or FeatureLogisticPredictor
    predictor = factory().fit(splits.train)
    val_acc = accuracy(predictor, splits.val)
    shift_acc = accuracy(predictor, splits.shift)
    shift_deg = round(max(0.0, val_acc - shift_acc), 4)

    baseline_val = accuracy(baseline_predictor, splits.val) if baseline_predictor is not None else 0.0
    # Majority-class baseline on the held-out split: a predictor that merely matches this
    # has learned the class prior, NOT action-outcome regularities. On a pass-skewed corpus
    # (strong-model traces) this baseline is already ~0.95, so clearing ``val_bar`` alone is
    # meaningless — promote requires STRICTLY BEATING it. Without this guard the canary
    # promotes a trivial always-positive solution on any pass-skewed data.
    majority_val = majority_class_accuracy(splits.val)
    kind = type(predictor).__name__

    if val_acc < val_bar:
        verdict, reason, promoted = "hold-below-bar", f"val accuracy {val_acc:.2f} < bar {val_bar}", False
    elif val_acc <= majority_val:
        verdict, reason, promoted = (
            "hold-at-majority-baseline",
            f"val accuracy {val_acc:.2f} does not beat the majority-class baseline "
            f"({majority_val:.2f}) — learned the class prior, not the action-outcome map",
            False,
        )
    elif shift_deg > max_shift_degradation:
        verdict, reason, promoted = (
            "hold-shift-degenerate",
            f"shift degradation {shift_deg:.2f} > {max_shift_degradation:.2f} — memorized, not generalized",
            False,
        )
    else:
        verdict, reason, promoted = "promote", "cleared held-out bar, beat majority baseline, and shift check", True

    return WorldModelReport(
        schema="sophia.verified_world_model.v1",
        candidate_only=True,
        level3_evidence=False,
        predictor_kind=kind,
        train_size=len(splits.train),
        val_size=len(splits.val),
        shift_size=len(splits.shift),
        val_accuracy=val_acc,
        shift_accuracy=shift_acc,
        shift_degradation=shift_deg,
        promoted=promoted,
        verdict=verdict,
        reason=reason,
        baseline_val_accuracy=baseline_val,
    )


__all__ = [
    "OutcomePair",
    "TrainValShift",
    "WorldModelReport",
    "FeatureLogisticPredictor",
    "accuracy",
    "make_splits",
    "train_verified_world_model",
]
