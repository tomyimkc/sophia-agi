#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for gate-invariant oracle (not a Gödel machine)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.godel_oracle import compute_solver_checked, evaluate_for_promotion  # noqa: E402


def _manifest_clean() -> dict:
    return {"contamination": {"eval": {"overlapCount": 0}}}


def test_clean_candidate_promotes_with_z3() -> None:
    traces = [{"metadata": {"sourceCitation": "https://example.org/src"}}]
    with patch("agent.invariant_suite.require_z3", side_effect=lambda fn, *a, **k: fn(*a, **k)):
        ok, bundle, path = evaluate_for_promotion(
            candidate_id="clean-test",
            protected_content_metrics=[
                {"suite": "religion", "contentBefore": 0.5, "contentAfter": 0.6},
            ],
            before_total=0.5,
            after_total=0.6,
            manifest=_manifest_clean(),
            traces=traces,
            tolerance=0.01,
            out_dir=Path(tempfile.mkdtemp()),
        )
    assert ok is True
    assert bundle["promote"] is True
    assert bundle["solverChecked"] is True
    assert path.exists()


def test_v5_religion_content_regression_blocks() -> None:
    with patch("agent.invariant_suite.require_z3", side_effect=lambda fn, *a, **k: fn(*a, **k)):
        ok, bundle, _path = evaluate_for_promotion(
            candidate_id="local-sophia-v5-full-religion-repair-mlx",
            protected_content_metrics=[
                {"suite": "religion", "contentBefore": 5 / 6, "contentAfter": 4 / 6},
                {"suite": "history", "contentBefore": 5 / 8, "contentAfter": 5 / 8},
            ],
            before_total=0.5,
            after_total=0.656,
            manifest=_manifest_clean(),
            traces=[{"metadata": {"sourceCitation": "https://example.org/src"}}],
            tolerance=0.01,
            out_dir=Path(tempfile.mkdtemp()),
        )
    assert ok is False
    assert "protected_floor_content" in bundle["breachingInvariants"]


def test_deterministic_bundle() -> None:
    kwargs = dict(
        candidate_id="det-test",
        protected_content_metrics=[
            {"suite": "religion", "contentBefore": 0.5, "contentAfter": 0.5},
        ],
        before_total=0.5,
        after_total=0.55,
        manifest=_manifest_clean(),
        traces=[{"metadata": {"sourceCitation": "https://example.org/src"}}],
        tolerance=0.01,
    )
    out = Path(tempfile.mkdtemp())
    with patch("agent.invariant_suite.require_z3", side_effect=lambda fn, *a, **k: fn(*a, **k)):
        _ok1, bundle1, _ = evaluate_for_promotion(**kwargs, out_dir=out / "a")
        _ok2, bundle2, _ = evaluate_for_promotion(**kwargs, out_dir=out / "b")
    for key in ("candidateId", "promote", "breachingInvariants", "invariants"):
        assert bundle1[key] == bundle2[key]
    assert "createdAt" not in json.dumps(bundle1)


def test_contamination_blocks() -> None:
    with patch("agent.invariant_suite.require_z3", side_effect=lambda fn, *a, **k: fn(*a, **k)):
        ok, bundle, _ = evaluate_for_promotion(
            candidate_id="dirty",
            protected_content_metrics=[
                {"suite": "religion", "contentBefore": 0.5, "contentAfter": 0.6},
            ],
            before_total=0.5,
            after_total=0.6,
            manifest={"contamination": {"eval": {"overlapCount": 1}}},
            traces=[{"metadata": {"sourceCitation": "https://example.org/src"}}],
            tolerance=0.01,
            out_dir=Path(tempfile.mkdtemp()),
        )
    assert ok is False
    assert "contamination_zero" in bundle["breachingInvariants"]


def test_solver_checked_false_blocks_without_z3() -> None:
    with patch("agent.formal_verifier.z3_available", return_value=False):
        ok, bundle, _ = evaluate_for_promotion(
            candidate_id="no-z3",
            protected_content_metrics=[
                {"suite": "religion", "contentBefore": 0.5, "contentAfter": 0.6},
            ],
            before_total=0.5,
            after_total=0.6,
            manifest=_manifest_clean(),
            traces=[{"metadata": {"sourceCitation": "https://example.org/src"}}],
            tolerance=0.01,
            out_dir=Path(tempfile.mkdtemp()),
        )
    assert ok is False
    assert bundle["solverChecked"] is False
    assert "solver_attestation" in bundle["breachingInvariants"]


def test_allow_fallback_promotes_without_z3() -> None:
    with patch("agent.formal_verifier.z3_available", return_value=False):
        ok, bundle, _ = evaluate_for_promotion(
            candidate_id="fallback-ok",
            protected_content_metrics=[
                {"suite": "religion", "contentBefore": 0.5, "contentAfter": 0.6},
            ],
            before_total=0.5,
            after_total=0.6,
            manifest=_manifest_clean(),
            traces=[{"metadata": {"sourceCitation": "https://example.org/src"}}],
            tolerance=0.01,
            out_dir=Path(tempfile.mkdtemp()),
            allow_fallback_proof=True,
        )
    assert ok is True
    assert bundle["solverChecked"] is False
    assert bundle.get("solverNotes") == ["fallback proof — not solver-checked"]


def test_compute_solver_checked_all_z3() -> None:
    inv = {
        "a": {"verdict": "accepted", "backend": "z3"},
        "b": {"verdict": "accepted", "backend": "z3"},
    }
    assert compute_solver_checked(inv) is True
    inv["b"]["backend"] = "fallback"
    assert compute_solver_checked(inv) is False


def main() -> int:
    test_clean_candidate_promotes_with_z3()
    test_v5_religion_content_regression_blocks()
    test_deterministic_bundle()
    test_contamination_blocks()
    test_solver_checked_false_blocks_without_z3()
    test_allow_fallback_promotes_without_z3()
    test_compute_solver_checked_all_z3()
    print("test_godel_oracle: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
