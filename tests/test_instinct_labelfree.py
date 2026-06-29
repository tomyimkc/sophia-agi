#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the label-free reflex comparison.

Pure stdlib, deterministic, offline (consumes stored v2 sample sets). Checks the signals
behave correctly, the recomputed exact signal reproduces the stored detector A, and the soft
'instability' signal beats exact self-consistency on DeepSeek with a CI that excludes chance.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_labelfree import (  # noqa: E402
    RESULTS,
    evaluate,
    main,
    sig_exact,
    sig_instability,
    sig_jaccard,
)


def test_self_test_entrypoint():
    assert main(["--self-test"]) == 0


def test_signals_on_known_inputs():
    unanimous = [frozenset({"a", "b"})] * 4
    assert sig_exact(unanimous) == 0.0
    assert sig_jaccard(unanimous) == 0.0
    assert sig_instability(unanimous) == 0.0
    split = [frozenset({"a", "b"}), frozenset({"a", "b"}), frozenset({"a"}), frozenset({"c"})]
    assert sig_exact(split) > 0.0
    assert sig_instability(split) > 0.0  # members disagree across samples


def test_recomputed_exact_matches_stored_A():
    ev = evaluate(RESULTS / "fusion_realmodel_deepseek.json")
    assert ev["exact_vs_storedA_drift"] < 1e-6


def test_instability_beats_exact_and_excludes_chance_on_deepseek():
    ev = evaluate(RESULTS / "fusion_realmodel_deepseek.json")
    assert ev["rows"]["instability"]["auc"] > ev["rows"]["exact"]["auc"]
    assert ev["rows"]["instability"]["auc_ci"][0] > 0.5   # reliably above chance
    # honest counter-point: exact's own CI still includes chance
    assert ev["rows"]["exact"]["auc_ci"][0] <= 0.5


def test_agreement_signals_useless_on_confident_wrong_model():
    # haiku is MORE self-consistent when wrong, so agreement-based AUC is near 0 (anti-predictive)
    ev = evaluate(RESULTS / "fusion_realmodel_llmhub-haiku.json")
    assert ev["rows"]["exact"]["auc"] < 0.3
    assert not ev["best_excludes_chance"]   # cannot be salvaged by a softer agreement signal


def test_all_aucs_in_range():
    for fname in ("fusion_realmodel_deepseek.json", "fusion_realmodel_llmhub-haiku.json"):
        ev = evaluate(RESULTS / fname)
        for r in ev["rows"].values():
            assert 0.0 <= r["auc"] <= 1.0 and r["auc_ci"][0] <= r["auc_ci"][1]


def test_cross_process_determinism():
    def once(seed_env: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=seed_env)
        return subprocess.run(
            [sys.executable, str(ROOT / "reasoning" / "instinct_labelfree.py"), "--run"],
            capture_output=True, text=True, env=env, check=True,
        ).stdout
    assert once("0") == once("314")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all instinct_labelfree tests passed")
