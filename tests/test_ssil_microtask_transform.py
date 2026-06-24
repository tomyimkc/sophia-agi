#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_plasticity import UpdateCandidate, evaluate_update  # noqa: E402
from agent.ssil_microtask_transform import (  # noqa: E402
    TransformSpec,
    demo_transform_report,
    measure_transform,
)


def test_good_transform_solves_and_promotes() -> None:
    good = TransformSpec.from_list(["strip", "upper", "replace: :_"])
    metrics, detail = measure_transform(good)
    assert detail["candidate"]["accuracy"] == 1.0
    cand = UpdateCandidate(id="t", kind="rule", verifier_artifacts=("heldout", "protected"), metrics=metrics)
    assert evaluate_update(cand, target_suite="transform_accuracy").verdict == "promote"


def test_identity_cannot_promote() -> None:
    metrics, _ = measure_transform(TransformSpec(ops=()))
    cand = UpdateCandidate(id="i", kind="rule", verifier_artifacts=("heldout", "protected"), metrics=metrics)
    assert evaluate_update(cand, target_suite="transform_accuracy").verdict in {"reject", "quarantine"}


def test_unsafe_ops_filtered() -> None:
    spec = TransformSpec.from_list(["upper", "__import__('os')", "replace:a:b"])
    assert "__import__('os')" not in spec.ops
    assert "upper" in spec.ops and "replace:a:b" in spec.ops


def test_demo_invariants() -> None:
    rep = demo_transform_report()
    assert all(rep["invariants"].values()), rep["invariants"]


def main() -> int:
    test_good_transform_solves_and_promotes()
    test_identity_cannot_promote()
    test_unsafe_ops_filtered()
    test_demo_invariants()
    print("test_ssil_microtask_transform: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
