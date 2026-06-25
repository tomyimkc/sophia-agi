#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for decidable numeric invariant suite."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.invariant_suite import (  # noqa: E402
    contamination_zero,
    no_total_regression,
    protected_floor_content,
    run_invariant_suite,
    solver_attestation,
)


def _manifest_clean() -> dict:
    return {"contamination": {"eval": {"overlapCount": 0}}}


def test_protected_floor_content_rejects_v5_religion_regression() -> None:
    metrics = [
        {"suite": "religion", "contentBefore": 5 / 6, "contentAfter": 4 / 6},
        {"suite": "history", "contentBefore": 5 / 8, "contentAfter": 5 / 8},
    ]
    result = protected_floor_content(metrics, tolerance=0.01)
    assert result["verdict"] == "rejected"


def test_contamination_overlap_blocks() -> None:
    bad = {"contamination": {"eval": {"overlapCount": 2}}}
    assert contamination_zero(bad)["verdict"] == "rejected"
    assert contamination_zero(_manifest_clean())["verdict"] == "accepted"


def test_missing_content_inputs_held() -> None:
    assert protected_floor_content([], tolerance=0.01)["verdict"] == "held"
    assert protected_floor_content(
        [{"suite": "religion", "contentBefore": 0.5}], tolerance=0.01
    )["verdict"] == "held"


def test_clean_candidate_all_accepted_with_z3() -> None:
    traces = [{"metadata": {"sourceCitation": "https://example.org/source"}}]
    with patch("agent.invariant_suite.require_z3", side_effect=lambda fn, *a, **k: fn(*a, **k)):
        results = run_invariant_suite(
            protected_content_metrics=[
                {"suite": "religion", "contentBefore": 0.5, "contentAfter": 0.6},
                {"suite": "history", "contentBefore": 0.6, "contentAfter": 0.6},
            ],
            before_total=0.5,
            after_total=0.6,
            manifest=_manifest_clean(),
            traces=traces,
            tolerance=0.01,
        )
    assert all(r["verdict"] == "accepted" for r in results.values())


def test_z3_absent_blocks_on_solver_attestation() -> None:
    from agent.invariant_suite import solver_attestation

    with patch("agent.formal_verifier.z3_available", return_value=False):
        result = solver_attestation()
    assert result["verdict"] == "held"
    assert result["status"] == "z3_unavailable"


def main() -> int:
    test_protected_floor_content_rejects_v5_religion_regression()
    test_contamination_overlap_blocks()
    test_missing_content_inputs_held()
    test_clean_candidate_all_accepted_with_z3()
    test_z3_absent_blocks_on_solver_attestation()
    print("test_invariant_suite: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
