#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the model↔harness closed-loop orchestrator (offline, deterministic).

These prove the loop's load-bearing properties without a GPU:

  * PLUMBING — the offline loop runs end-to-end and restores the harness run dir.
  * NON-DEGENERACY — a promoted candidate whose post-uplift is negative HALTS
    LOUD and the spec is rolled back (defense-in-depth over the plasticity gate,
    which checks pass-rate delta, not the harness-vs-bare relationship).
  * GATE HONORED — a candidate whose harness pass-rate regresses is rejected by
    the plasticity gate and never promoted.
  * PROMOTION — a candidate that holds rate (delta >= floor) is promoted and the
    spec advances.
  * SATURATION — uplift collapsing to ~0 AFTER a promotion is flagged success.
  * NO-OVERCLAIM — every report carries candidateOnly=True, level3Evidence=False.

Uplift is a function of (client text) × (verifier); both are controlled here via
a scripted client keyed by spec + a mustInclude marker check in the suite. This
lets each test script deterministic bare-vs-harness outcomes.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import closed_loop as cl  # noqa: E402
from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402

# Suite whose verifier is "does the output contain the marker 'Decision'".
# A scripted client controls bare-vs-harness by emitting the marker or not.
_SUITE = [
    {"id": "d1", "goal": "Decide the next step.", "mode": "advisor", "mustInclude": ["Decision"]},
    {"id": "d2", "goal": "Decide whether to promote.", "mode": "advisor", "mustInclude": ["Decision"]},
]


class _SpecClient:
    """Returns a fixed ModelResult per spec string. ModelClient duck-types: the
    loop only calls .generate(system, user)."""

    def __init__(self, text: str):
        self._text = text

    def generate(self, system: str, user: str):
        return m.ModelResult(text=self._text, provider="stub", model="stub", ok=True)


def _factory(spec_text: dict[str, str], default: str):
    def make(spec: str) -> _SpecClient:
        return _SpecClient(spec_text.get(spec, default))

    return make


def _advancing_train(to_spec: str):
    """TrainStep that always 'trains' into `to_spec` (simulates a GPU run)."""

    def step(_cycle: int, _pairs: list, _current: str) -> cl.TrainOutcome:
        return cl.TrainOutcome(new_spec=to_spec, ran=True, artifact=to_spec, notes="scripted")

    return step


# Texts that deterministically pass/fail the real epistemic gate (verified
# empirically: this exact _GOOD string passes check_response; _GOOD-without
# the discipline/中文 markers, _BAD, fails). Using the real gate keeps these
# tests honest about what the loop actually gates on.
_GOOD = "[mock:m] Analysis.\nDecision: proceed (mock). source discipline noted.\n中文摘要: 模拟回答。"
_BAD = "i have no idea what to do next without the marker"


def test_offline_loop_runs_and_restores_runs_dir() -> None:
    """Noop-train loop completes all cycles, restores h.RUNS_DIR, promotes
    nothing (nothing to gate), stays non-degenerate."""
    saved = h.RUNS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        report = cl.run_closed_loop(
            _SUITE, suite_name="t", make_client=_factory({}, _GOOD),
            initial_spec="base", train_step=cl.noop_train_step,
            runs_root=Path(tmp), max_cycles=2,
        )
    assert h.RUNS_DIR == saved, "loop leaked its per-cycle RUNS_DIR override"
    assert len(report.cycles) == 2
    assert report.non_degenerate is True
    assert report.promoted_any is False
    assert all(c.promotion_verdict == "no-candidate" for c in report.cycles)
    assert report.candidateOnly is True and report.level3Evidence is False


def test_candidate_holding_rate_is_promoted() -> None:
    """A candidate that holds the same pass rate (delta 0 >= floor 0) is promoted
    and the spec advances."""
    with tempfile.TemporaryDirectory() as tmp:
        report = cl.run_closed_loop(
            _SUITE, suite_name="t",
            make_client=_factory({}, _GOOD),  # base + candidate both _GOOD
            initial_spec="base",
            train_step=_advancing_train("cand-v1"),
            runs_root=Path(tmp), max_cycles=2, min_target_delta=0.0,
        )
    assert report.promoted_any is True
    assert report.final_model_spec == "cand-v1"
    assert report.cycles[0].model_advanced is True
    assert report.cycles[0].promotion_verdict == "promote"


def test_gate_blocks_pass_rate_regression() -> None:
    """A candidate whose target-suite pass-rate DROPS vs baseline is blocked by
    the plasticity gate (floor-miss => quarantine; protected-suite regression or
    contamination => reject) and never promoted, even though the train step
    'ran'. base=_GOOD (rate 1.0), cand=_BAD (rate 0.0) => delta -1.0 < floor 0.

    Either verdict blocks promotion; what matters is the spec never advances."""
    with tempfile.TemporaryDirectory() as tmp:
        report = cl.run_closed_loop(
            _SUITE, suite_name="t",
            make_client=_factory({"cand-v1": _BAD}, default=_GOOD),
            initial_spec="base",
            train_step=_advancing_train("cand-v1"),
            runs_root=Path(tmp), max_cycles=2, min_target_delta=0.0,
        )
    assert report.final_model_spec == "base"  # never advanced
    assert report.cycles[0].model_advanced is False
    assert report.cycles[0].promotion_verdict in {"reject", "quarantine"}
    assert any("floor" in r.lower() or "regression" in r.lower()
               for r in report.cycles[0].promotion_reasons)


def test_non_degeneracy_halts_on_negative_post_uplift() -> None:
    """DEFENSE-IN-DEPTH over the plasticity gate: even if the gate cleared, a
    promoted candidate whose *uplift* (harness − bare) goes negative HALTS the
    loop and rolls back to the baseline spec.

    We force the gate to promote (monkeypatch evaluate_update) so the gate is
    NOT what protects us here — the non-degeneracy wall is. The candidate uses
    a split client: first generation (bare) passes the gate, subsequent ones
    (the harness loop) fail => bare-pass / harness-fail => negative uplift.
    """
    from agent import continual_plasticity as cp

    class _SplitClient:
        """First generate() = bare (passes gate); subsequent = harness (fails)."""

        def __init__(self):
            self._calls = 0

        def generate(self, system, user):
            self._calls += 1
            txt = _GOOD if self._calls == 1 else _BAD
            return m.ModelResult(text=txt, provider="stub", model="stub", ok=True)

    def make(_spec):
        return _SplitClient()  # fresh counter per measure_uplift condition

    real_evaluate = cp.evaluate_update
    cp.evaluate_update = lambda c, **k: cp.PromotionDecision(  # type: ignore[assignment]
        candidate_id=c.id, verdict="promote", reasons=("forced",), metrics={"forced": True},
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            report = cl.run_closed_loop(
                _SUITE, suite_name="t", make_client=make,
                initial_spec="base",
                train_step=_advancing_train("cand-v1"),
                runs_root=Path(tmp), max_cycles=2, min_target_delta=0.0,
            )
    finally:
        cp.evaluate_update = real_evaluate  # type: ignore[assignment]

    assert report.non_degenerate is False, "negative post-uplift must trip the wall"
    assert report.halted_early is True
    assert report.final_model_spec == "base"  # rolled back, not advanced
    assert "negative uplift" in report.halt_reason


def test_saturation_flagged_after_promotion() -> None:
    """A promoted model whose final uplift is ~0 is SATURATED (success)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = cl.run_closed_loop(
            _SUITE, suite_name="t",
            make_client=_factory({}, _GOOD),  # identical behavior pre/post
            initial_spec="base",
            train_step=_advancing_train("cand-v1"),
            runs_root=Path(tmp), max_cycles=2, min_target_delta=0.0,
            saturation_eps=0.02,
        )
    assert report.promoted_any is True
    assert report.saturated is True
    assert report.non_degenerate is True
    assert "SATURATED" in report.to_dict()["interpretation"]


def test_report_payload_marks_candidate_not_level3() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        report = cl.run_closed_loop(
            _SUITE, suite_name="t", make_client=_factory({}, _GOOD),
            initial_spec="base", train_step=cl.noop_train_step,
            runs_root=Path(tmp), max_cycles=2,
        )
    payload = report.to_dict()
    assert payload["candidateOnly"] is True
    assert payload["level3Evidence"] is False


def main() -> int:
    test_offline_loop_runs_and_restores_runs_dir()
    test_candidate_holding_rate_is_promoted()
    test_gate_blocks_pass_rate_regression()
    test_non_degeneracy_halts_on_negative_post_uplift()
    test_saturation_flagged_after_promotion()
    test_report_payload_marks_candidate_not_level3()
    print("test_closed_loop: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
