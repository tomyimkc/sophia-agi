#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_plasticity import EvalMetric, UpdateCandidate, demo_plasticity_report, evaluate_update  # noqa: E402


def test_clean_update_promotes() -> None:
    c = UpdateCandidate(
        id="u1", kind="skill", verifier_artifacts=("heldout", "anti-regression"),
        metrics=(EvalMetric("target", 0.5, 0.6), EvalMetric("protected", 0.9, 0.9, protected=True)),
    )
    assert evaluate_update(c, target_suite="target").verdict == "promote"


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
    test_regression_rejects()
    test_plasticity_demo_invariants()
    print("test_continual_plasticity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
