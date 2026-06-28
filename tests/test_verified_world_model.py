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


def test_majority_class_accuracy_helper() -> None:
    # All-positive => always-predict-1 baseline = 1.0; balanced => 0.5.
    assert vwm.majority_class_accuracy([("s", "a", 1)] * 9 + [("s", "a", 0)]) == 0.9
    assert vwm.majority_class_accuracy([("s", "a", 1), ("s", "a", 0)]) == 0.5
    assert vwm.majority_class_accuracy([]) == 0.0


def test_pass_skewed_corpus_does_not_spuriously_promote() -> None:
    """A pass-skewed corpus (strong-model traces, ~95% positive) must NOT promote a
    predictor that merely matches the majority class.

    Regression guard for the canary defect found on the 2026-06-26 real-DeepSeek refire:
    val/shift were all-positive (positive_rate 1.0), so val_accuracy 1.0 == the
    always-predict-positive baseline, yet the old gate promoted. The fix requires the
    predictor to STRICTLY BEAT the majority-class baseline before promote.
    """
    # Pass-skewed, with ALL negatives in train-only states (mirrors the real refire where
    # the 2 failures both fell in train, leaving val/shift all-positive). The negative
    # states ("fail_state_*") are unique so the in-distribution train/val carve-out keeps
    # them in train; the all-positive "ok_*" states populate val.
    rows = (
        [("ok_alpha", "crossref", 1)] * 12
        + [("ok_beta", "crossref", 1)] * 12
        + [("ok_gamma", "crossref", 1)] * 12
        + [("fail_state_one", "crossref", 0), ("fail_state_two", "crossref", 0)]
        + [("shifted_state", "crossref", 1)] * 4
    )
    report = vwm.train_verified_world_model(rows, val_bar=0.65, seed=0)
    # The held-out val split is all-positive, so its majority-class baseline is 1.0 —
    # UNBEATABLE. The predictor's val_accuracy can at most match it (1.0), never beat it,
    # so it must HOLD even though 1.0 clears the 0.65 bar.
    assert report.verdict == "hold-at-majority-baseline"
    assert report.promoted is False
    splits = vwm.make_splits(rows, seed=0)
    assert report.val_accuracy == vwm.majority_class_accuracy(splits.val)
    assert "majority-class baseline" in report.reason


def main() -> int:
    test_make_splits_keeps_shift_distribution_separate()
    test_learnable_signal_is_promoted()
    test_shift_degenerate_model_is_held_not_promoted()
    test_below_bar_is_held()
    test_report_discipline_fields()
    test_injected_predictor_seam()
    test_majority_class_accuracy_helper()
    test_pass_skewed_corpus_does_not_spuriously_promote()
    print("test_verified_world_model: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
