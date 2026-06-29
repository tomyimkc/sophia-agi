#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the T3 calibration-verifier eval machinery. Offline, stdlib only, no torch.

Asserts: AUROC is correct on separable/reversed/tied inputs and None when a class is absent;
ECE is ~0 for a perfectly-calibrated set and large for a confidently-wrong set; the mock run
is deterministic and returns NO-GO with the pre-registered critical failures; gate_verdict only
says GO when every pillar is met; and emit_pending writes a not_run / NO-GO artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_calibration_verifier_eval import (  # noqa: E402
    auroc,
    build_pending_artifact,
    ece,
    gate_verdict,
    run_mock,
)


def test_auroc_separable_and_reversed() -> None:
    # correct (label 1) all score higher than incorrect (label 0) -> AUROC 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    # perfectly reversed -> 0.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    # all tied -> 0.5
    assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5


def test_auroc_none_when_one_class_absent() -> None:
    assert auroc([0.9, 0.8], [1, 1]) is None
    assert auroc([0.1, 0.2], [0, 0]) is None


def test_ece_calibrated_vs_miscalibrated() -> None:
    # confidently right (prob 1.0, all correct) and confidently sure-wrong-is-wrong (prob 0, all wrong)
    well = ece([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0])
    assert well is not None and well < 0.01, well
    # confidently wrong: prob 1.0 but all incorrect -> ECE near 1.0
    bad = ece([1.0, 1.0, 1.0], [0, 0, 0])
    assert bad is not None and bad > 0.9, bad


def test_mock_run_is_deterministic_and_no_go() -> None:
    a = run_mock(n=200, seed=0)
    b = run_mock(n=200, seed=0)
    assert a == b, "mock run must be deterministic for a fixed seed"
    assert a["verdict"] == "NO-GO", a
    # the offline pillars that can never pass with synthetic data must be named
    joined = " ".join(a["criticalFailures"])
    assert "no_real_corpus" in joined and "labels_not_2family" in joined and "no_leakage_audit" in joined, a
    # the math actually ran
    assert a["arms"]["trace-feature-verifier"]["auroc"] is not None, a
    assert isinstance(a["deltaAUROCCI95"], list) and len(a["deltaAUROCCI95"]) == 2, a


def test_gate_verdict_go_only_when_all_pillars_met() -> None:
    go = gate_verdict(real_corpus=True, judge_families=2, leakage_audited=True,
                      delta_auroc=0.08, delta_auroc_ci=[0.02, 0.13], delta_ece=-0.01, base_sizes=3)
    assert go["verdict"] == "GO" and go["go"] is True, go
    # drop a single pillar -> NO-GO
    no = gate_verdict(real_corpus=True, judge_families=2, leakage_audited=True,
                      delta_auroc=0.08, delta_auroc_ci=[0.02, 0.13], delta_ece=+0.02, base_sizes=3)
    assert no["verdict"] == "NO-GO", no
    assert any("calibration_guardrail" in f for f in no["criticalFailures"]), no


def test_pending_artifact_shape() -> None:
    art = build_pending_artifact()
    assert art["status"] == "not_run" and art["go"] is False and art["canClaimAGI"] is False, art
    assert art["verdict"] == "NO-GO", art
    assert art["preregistration"].endswith("measurement_spec.json"), art


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
