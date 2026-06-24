#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_plasticity import (  # noqa: E402
    EvalMetric,
    RetentionEvidence,
    UpdateCandidate,
    demo_plasticity_report,
    evaluate_update,
)


def test_clean_update_promotes() -> None:
    c = UpdateCandidate(
        id="u1", kind="skill", verifier_artifacts=("heldout", "anti-regression"),
        metrics=(EvalMetric("target", 0.5, 0.6), EvalMetric("protected", 0.9, 0.9, protected=True)),
    )
    assert evaluate_update(c, target_suite="target").verdict == "promote"


def _improving() -> UpdateCandidate:
    return UpdateCandidate(
        id="u-ret", kind="lora_adapter", verifier_artifacts=("heldout", "anti-regression"),
        metrics=(EvalMetric("target", 0.5, 0.75), EvalMetric("protected", 0.9, 0.9, protected=True)),
    )


def test_catastrophic_forgetting_rejects() -> None:
    """A target-suite gain bought by forgetting the old task is a HARD reject."""
    forgot = RetentionEvidence(old_benchmark_delta_pct=-50.0, passing_signal=False, evaluable="evaluated")
    d = evaluate_update(_improving(), target_suite="target", retention=forgot)
    assert d.verdict == "reject"
    assert any("retention regression" in r for r in d.reasons)
    assert d.metrics["retention"]["forgetting"] is True


def test_retained_old_task_still_promotes() -> None:
    """Old task within tolerance does not block an otherwise-clean promotion."""
    held = RetentionEvidence(old_benchmark_delta_pct=-2.0, passing_signal=True, evaluable="evaluated")
    d = evaluate_update(_improving(), target_suite="target", retention=held)
    assert d.verdict == "promote", d.reasons


def test_missing_retention_is_backward_compatible() -> None:
    """No retention evidence -> unchanged behavior (does not silently reject)."""
    assert evaluate_update(_improving(), target_suite="target").verdict == "promote"


def test_required_but_unverifiable_retention_quarantines() -> None:
    """When retention is required, an unverifiable signal quarantines rather than promotes."""
    no_baseline = RetentionEvidence(old_benchmark_delta_pct=None, evaluable="requested-but-no-baseline")
    d = evaluate_update(_improving(), target_suite="target", retention=no_baseline, require_retention=True)
    assert d.verdict == "quarantine"
    assert any("retention evidence required" in r for r in d.reasons)


def test_regression_rejects() -> None:
    c = UpdateCandidate(
        id="u2", kind="skill", verifier_artifacts=("heldout", "anti-regression"),
        metrics=(EvalMetric("target", 0.5, 0.7), EvalMetric("protected", 0.9, 0.84, protected=True)),
    )
    d = evaluate_update(c, target_suite="target")
    assert d.verdict == "reject"
    assert any("protected regression" in r for r in d.reasons)


def test_plasticity_demo_invariants() -> None:
    rep = demo_plasticity_report()
    assert all(rep["invariants"].values())


def main() -> int:
    test_clean_update_promotes()
    test_catastrophic_forgetting_rejects()
    test_retained_old_task_still_promotes()
    test_missing_retention_is_backward_compatible()
    test_required_but_unverifiable_retention_quarantines()
    test_regression_rejects()
    test_plasticity_demo_invariants()
    print("test_continual_plasticity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
