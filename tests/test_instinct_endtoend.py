#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the end-to-end instinct outcome simulator.

Pure stdlib, deterministic, offline (consumes the committed real-model artifacts). Checks:
(E1) a usable detector cuts confident-wrong below commit and late; (E2) correctness does not
fall below commit; (E3) a blind detector yields ~no change (uplift is gated by detection);
(E4) confident-wrong falls monotonically with detector TPR; plus cross-process determinism.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_endtoend import (  # noqa: E402
    OperatingPoint,
    main,
    profile_from_artifact,
    run_experiment,
    simulate,
    tpr_sweep,
)

RESULTS = ROOT / "reasoning" / "results"


def test_self_test_entrypoint():
    assert main(["--self-test"]) == 0


def test_profiles_from_real_artifacts():
    ds = profile_from_artifact(RESULTS / "fusion_realmodel_deepseek.json")
    hk = profile_from_artifact(RESULTS / "fusion_realmodel_llmhub-haiku.json")
    assert 0.4 < ds.base_error < 0.7 and ds.tpr > 0.5   # DeepSeek: usable detector
    # haiku: still fails the task, but B2 now makes its under-abstentions detectable
    assert hk.base_error >= 0.9 and hk.tpr > 0.8


def test_e1_safety_win_with_usable_detector():
    op = profile_from_artifact(RESULTS / "fusion_realmodel_deepseek.json")
    pol = simulate(op, seed=1234)
    assert pol["instinct"].wrong_asserted < pol["commit"].wrong_asserted
    assert pol["instinct"].wrong_asserted < pol["late"].wrong_asserted


def test_e2_recovery_not_just_abstention():
    op = profile_from_artifact(RESULTS / "fusion_realmodel_deepseek.json")
    pol = simulate(op, seed=1234)
    assert pol["instinct"].correct >= pol["commit"].correct - 0.01


def test_e3_haiku_rescued_by_failclose():
    # With B2 in the bus, haiku's confident-wrong collapses to escalation (not correctness).
    op = profile_from_artifact(RESULTS / "fusion_realmodel_llmhub-haiku.json")
    pol = simulate(op, seed=1234)
    assert pol["commit"].wrong_asserted > 0.8          # baseline: confidently wrong
    assert pol["instinct"].wrong_asserted < 0.1        # eliminated
    assert pol["instinct"].escalate > 0.8              # converted to fail-closed escalate
    assert pol["instinct"].correct < 0.2               # task still not solved


def test_blind_detector_synthetic_no_uplift():
    # The principle still holds for a genuinely blind detector (TPR 0): instinct ≈ commit.
    pol = simulate(OperatingPoint("blind", base_error=0.5, tpr=0.0, fpr=0.0), seed=1234)
    assert abs(pol["instinct"].wrong_asserted - pol["commit"].wrong_asserted) < 0.03


def test_e4_confident_wrong_monotone_in_tpr():
    sweep = tpr_sweep(base_error=0.54, fpr=0.26, seed=1234)
    wa = [r["wrong_asserted"] for r in sweep]
    assert all(wa[i] >= wa[i + 1] - 0.01 for i in range(len(wa) - 1)), wa
    assert wa[0] - wa[-1] > 0.2
    # correctness should rise as recall rises
    cr = [r["correct"] for r in sweep]
    assert cr[-1] > cr[0]


def test_escalate_is_failclosed_not_wrong():
    # A perfect-recall, high-base, blind-to-clean profile converts errors to escalate, not wrong.
    op = OperatingPoint("x", base_error=1.0, tpr=1.0, fpr=0.0)
    pol = simulate(op, seed=1234)
    assert pol["instinct"].escalate > 0.9
    assert pol["instinct"].wrong_asserted < 0.05


def test_cross_process_determinism():
    def once(seed_env: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=seed_env)
        return subprocess.run(
            [sys.executable, str(ROOT / "reasoning" / "instinct_endtoend.py"), "--self-test"],
            capture_output=True, text=True, env=env, check=True,
        ).stdout
    assert once("0") == once("99")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all instinct_endtoend tests passed")
