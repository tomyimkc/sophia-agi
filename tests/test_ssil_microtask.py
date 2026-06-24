#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_plasticity import UpdateCandidate, evaluate_update  # noqa: E402
from agent.ssil_microtask import (  # noqa: E402
    PolicySpec,
    baseline_spec,
    demo_microtask_report,
    load_cases,
    measure_policy,
    train_feature_summary,
)


def test_dataset_loads_and_splits() -> None:
    train, test = load_cases("train"), load_cases("test")
    assert len(train) >= 20 and len(test) >= 30
    assert all(c["gold"] in {"answer", "abstain"} for c in train + test)


def test_train_summary_hides_test_labels() -> None:
    s = train_feature_summary()
    assert "trainSize" in s and "qualityRange" in s
    assert "gold" not in json.dumps(s)  # the proposer never receives labels


def test_good_policy_beats_baseline() -> None:
    metrics, detail = measure_policy(PolicySpec(min_sources=2, min_quality=0.6))
    by = {m.suite: m for m in metrics}
    assert by["routing_accuracy"].after > by["routing_accuracy"].before
    assert by["answer_recall"].protected is True
    assert detail["candidate"]["accuracy"] > detail["baseline"]["accuracy"]


def test_good_policy_promotes_through_g4() -> None:
    metrics, _ = measure_policy(PolicySpec(min_sources=2, min_quality=0.6))
    cand = UpdateCandidate(id="p", kind="rule", verifier_artifacts=("heldout-measured", "protected-recall"), metrics=metrics)
    assert evaluate_update(cand, target_suite="routing_accuracy").verdict == "promote"


def test_degenerate_abstain_cannot_promote() -> None:
    metrics, _ = measure_policy(PolicySpec(min_sources=99, min_quality=1.1))  # never answers
    cand = UpdateCandidate(id="d", kind="rule", verifier_artifacts=("heldout-measured", "protected-recall"), metrics=metrics)
    assert evaluate_update(cand, target_suite="routing_accuracy").verdict in {"reject", "quarantine"}


def test_baseline_is_weak() -> None:
    metrics, detail = measure_policy(baseline_spec())
    # Baseline (answer-everything) should not clear the improvement floor over itself.
    assert detail["baseline"]["accuracy"] == detail["candidate"]["accuracy"]


def test_demo_invariants() -> None:
    rep = demo_microtask_report()
    assert all(rep["invariants"].values()), rep["invariants"]


def main() -> int:
    test_dataset_loads_and_splits()
    test_train_summary_hides_test_labels()
    test_good_policy_beats_baseline()
    test_good_policy_promotes_through_g4()
    test_degenerate_abstain_cannot_promote()
    test_baseline_is_weak()
    test_demo_invariants()
    print("test_ssil_microtask: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
