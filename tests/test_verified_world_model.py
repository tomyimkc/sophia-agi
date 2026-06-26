#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the verified world-model training scaffold (offline, deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verified_world_model as vwm  # noqa: E402


def _learnable_traces() -> list[vwm.OutcomePair]:
    """A signal the feature logistic CAN learn: action 'crossref' succeeds on
    in-distribution states; action 'guess' fails. Repeated so train/val both see it."""
    rows = []
    for _ in range(8):
        rows += [
            ("claim_held", "crossref", 1),
            ("claim_held", "guess", 0),
            ("one_source", "crossref", 1),
            ("one_source", "guess", 0),
        ]
    return rows


def test_make_splits_keeps_shift_distribution_separate() -> None:
    """shift_states traces go to shift; the rest split train/val. No overlap."""
    traces = _learnable_traces() + [("shifted_state", "crossref", 1)] * 5
    splits = vwm.make_splits(traces, shift_states={"shifted_state"})
    assert all(s != "shifted_state" for s, _, _ in splits.train + splits.val)
    assert all(s == "shifted_state" for s, _, _ in splits.shift)
    assert splits.train and splits.val and splits.shift


def test_learnable_signal_is_promoted() -> None:
    """A signal the predictor can learn clears the held-out bar and (with matching
    shift) is promoted."""
    traces = _learnable_traces()
    # shift uses the SAME action signal but a new state, so it generalizes
    shift = [("novel_state", "crossref", 1)] * 6 + [("novel_state", "guess", 0)] * 6
    splits = vwm.TrainValShift(train=[], val=[], shift=[])
    # rebuild proper splits then graft the shift
    base = vwm.make_splits(traces, seed=0)
    splits = vwm.TrainValShift(train=base.train, val=base.val, shift=shift)
    report = vwm.train_verified_world_model(traces, splits=splits, val_bar=0.7)
    assert report.promoted is True, report.reason
    assert report.verdict == "promote"
    assert report.val_accuracy >= 0.7


def test_shift_degenerate_model_is_held_not_promoted() -> None:
    """The core research risk: a predictor that aces held-out but collapses under
    shift has MEMORIZED, not generalized => held as shift-degenerate, never promoted
    (so the planner is not misled)."""
    traces = _learnable_traces()
    # shift REVERSES the action->outcome mapping the model learned => it will fail
    shift = [("novel_state", "crossref", 0)] * 6 + [("novel_state", "guess", 1)] * 6
    base = vwm.make_splits(traces, seed=0)
    splits = vwm.TrainValShift(train=base.train, val=base.val, shift=shift)
    report = vwm.train_verified_world_model(traces, splits=splits, val_bar=0.5, max_shift_degradation=0.15)
    assert report.promoted is False
    assert report.verdict == "hold-shift-degenerate"
    assert report.shift_degradation > 0.15


def test_below_bar_is_held() -> None:
    """An unlearnable signal (label independent of features) => val < bar => held."""
    noise = [("s", "a", i % 2) for i in range(40)]  # no learnable structure
    splits = vwm.make_splits(noise, seed=1)
    report = vwm.train_verified_world_model(noise, splits=splits, val_bar=0.7)
    assert report.promoted is False
    assert report.verdict == "hold-below-bar"


def test_report_discipline_fields() -> None:
    """No-overclaim: candidate + not Level-3 regardless of outcome."""
    report = vwm.train_verified_world_model(_learnable_traces(), val_bar=0.0, max_shift_degradation=1.0)
    assert report.candidate_only is True
    assert report.level3_evidence is False
    d = report.to_dict()
    assert d["candidateOnly"] is True and d["level3Evidence"] is False


def test_injected_predictor_seam() -> None:
    """A custom predictor factory is used (the live torch seam). Here a constant
    predictor that always predicts the majority class — proves injection works."""

    class ConstantPredictor:
        def fit(self, pairs):
            return self

        def predict(self, s, a):
            return 0.5  # always uncertain => ~50% accuracy

    report = vwm.train_verified_world_model(
        _learnable_traces(), predictor_factory=ConstantPredictor, val_bar=0.6,
    )
    assert report.predictor_kind == "ConstantPredictor"


def main() -> int:
    test_make_splits_keeps_shift_distribution_separate()
    test_learnable_signal_is_promoted()
    test_shift_degenerate_model_is_held_not_promoted()
    test_below_bar_is_held()
    test_report_discipline_fields()
    test_injected_predictor_seam()
    print("test_verified_world_model: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
