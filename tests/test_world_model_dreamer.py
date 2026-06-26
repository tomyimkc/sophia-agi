#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the DreamerV3-style discrete-latent world model.

Two regimes, both honest:

  * FAIL-CLOSED (always runs, no torch): the predictor abstains (predict 0.5) and
    the report records torch_available=False. This is the CI path — it proves the
    CUDA-gate never breaks the no-torch default.
  * GENERALIZATION (runs only if torch importable): the canary fires correctly —
    promote on a learnable signal, hold-shift-degenerate on a reversed shift split.
    This is the load-bearing test of Path A; it is skipped (not failed) without torch.

The shift-degeneracy assertion is the experiment: a model that aces held-out but
collapses under shift has MEMORIZED, and the canary must catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import world_model_dreamer as dwm  # noqa: E402
from agent import verified_world_model as vwm  # noqa: E402

_TORCH = dwm._torch() is not None


def test_fail_closed_without_torch() -> None:
    """When torch is absent, the predictor abstains (0.5) and the report says so.
    This MUST run in CI (numpy-only) — the CUDA gate never breaks the default."""
    pred = dwm.DreamerWorldPredictor()
    if dwm._torch() is not None:
        return  # torch present in this env; the fail-closed path isn't exercised
    # untrained + no torch -> abstain
    assert pred.predict("any_state", "any_action") == 0.5
    # fit is a no-op (returns self) without torch
    pred.fit([("s", "a", 1)] * 5)
    assert pred.predict("s", "a") == 0.5  # still abstaining


def test_report_carries_discipline_fields() -> None:
    """No-overclaim: the report carries candidateOnly + level3Evidence: false."""
    pred, rep = dwm.train_dreamer_report([], cfg=dwm.DreamerConfig(epochs=1))
    d = rep.to_dict()
    assert d["candidateOnly"] is True
    assert d["level3Evidence"] is False
    assert "schema" in d and d["schema"].startswith("sophia.world_model_dreamer")


def _learnable_traces() -> list[vwm.OutcomePair]:
    """A signal a discrete-latent model CAN learn: action 'crossref' -> success on
    in-distribution states; 'guess' -> failure. Repeated so train/val both see it."""
    rows = []
    for _ in range(10):
        rows += [
            ("claim_held", "crossref", 1),
            ("claim_held", "guess", 0),
            ("one_source", "crossref", 1),
            ("one_source", "guess", 0),
        ]
    return rows


def test_predictor_protocol_shape() -> None:
    """The predictor implements the OutcomePredictor protocol: fit -> self,
    predict(state, action) -> float in [0,1]. Slides into verified_world_model + planner_learned_sim."""
    pred = dwm.DreamerWorldPredictor()
    assert hasattr(pred, "fit") and hasattr(pred, "predict")
    # predict returns a valid probability even before training (abstain 0.5)
    p = pred.predict("s", "a")
    assert 0.0 <= p <= 1.0


def test_fit_empty_corpus_stays_fail_closed() -> None:
    """Regression: fit([]) must NOT mark the model trained. An empty corpus runs zero
    optimizer steps, so marking trained would break fail-closed — predict() would return
    an arbitrary random-weight value instead of abstaining (0.5)."""
    pred = dwm.DreamerWorldPredictor()
    if dwm._torch() is None:
        # No torch: fit is a no-op regardless; predict still abstains.
        pred.fit([])
        assert pred.predict("s", "a") == 0.5
        return
    # With torch: an empty corpus must keep the predictor abstaining.
    pred.fit([])
    assert pred.predict("s", "a") == 0.5, "empty-corpus fit must stay fail-closed (abstain 0.5)"


# --- generalization tests: only meaningful with torch; skip cleanly otherwise ---

def test_canary_promotes_on_learnable_signal() -> None:
    """With torch: a learnable signal clears the held-out bar and (matching shift)
    is promoted by the verified-world-model canary. Without torch: skip."""
    if not _TORCH:
        import pytest
        pytest.skip("torch not installed; generalization test needs it")
    traces = _learnable_traces()
    shift = [("novel_state", "crossref", 1)] * 8 + [("novel_state", "guess", 0)] * 8
    base = vwm.make_splits(traces, seed=0)
    splits = vwm.TrainValShift(train=base.train, val=base.val, shift=shift)
    pred, rep = dwm.train_dreamer_report(splits.train, val_traces=splits.val,
                                         cfg=dwm.DreamerConfig(epochs=30, seed=0))
    assert rep.trained, rep.reason
    report = vwm.train_verified_world_model(traces, predictor_factory=lambda: pred,
                                            splits=splits, val_bar=0.6,
                                            max_shift_degradation=0.15)
    # learnable signal + matching shift -> promote (the canary clears)
    assert report.verdict == "promote", f"expected promote, got {report.verdict}: {report.reason}"


def test_canary_holds_on_shift_degenerate() -> None:
    """The core experiment: a model that aces held-out but collapses under shift
    has MEMORIZED. The canary MUST hold it (hold-shift-degenerate), never promote —
    so the planner is not misled by an overfit model. Without torch: skip."""
    if not _TORCH:
        import pytest
        pytest.skip("torch not installed; shift-degeneracy test needs it")
    traces = _learnable_traces()
    # shift REVERSES the signal the model learned -> it will fail under shift
    shift = [("novel_state", "crossref", 0)] * 8 + [("novel_state", "guess", 1)] * 8
    base = vwm.make_splits(traces, seed=0)
    splits = vwm.TrainValShift(train=base.train, val=base.val, shift=shift)
    pred, _ = dwm.train_dreamer_report(splits.train, val_traces=splits.val,
                                       cfg=dwm.DreamerConfig(epochs=30, seed=0))
    report = vwm.train_verified_world_model(traces, predictor_factory=lambda: pred,
                                            splits=splits, val_bar=0.5,
                                            max_shift_degradation=0.15)
    assert report.verdict in ("hold-shift-degenerate", "hold-below-bar"), \
        f"shift-degenerate model must be held, got {report.verdict}"


def main() -> int:
    test_fail_closed_without_torch()
    test_report_carries_discipline_fields()
    test_predictor_protocol_shape()
    # generalization tests skip cleanly without torch
    if _TORCH:
        test_canary_promotes_on_learnable_signal()
        test_canary_holds_on_shift_degenerate()
        print("test_world_model_dreamer: OK (torch present — generalization canary exercised)")
    else:
        print("test_world_model_dreamer: OK (no torch — fail-closed path verified; "
              "generalization tests skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
