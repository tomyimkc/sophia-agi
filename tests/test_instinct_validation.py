#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the fusion-validation layer (cross-val weights + bootstrap CIs).

Pure stdlib, deterministic, offline (consumes the committed v2 artifacts). Checks: LOO-CV
removes in-sample optimism; bootstrap CIs are well-formed; the structural verifiers (B/B2)
outrank the label-free reflex (A), whose CI still includes chance; low-clean models are flagged.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_validation import (  # noqa: E402
    RESULTS,
    bootstrap_ci,
    evaluate,
    insample_qw_fused,
    load_scores,
    loo_qw_fused,
    main,
)


def test_self_test_entrypoint():
    assert main(["--self-test"]) == 0


def test_loocv_not_more_optimistic_than_insample():
    scores, labels = load_scores(RESULTS / "fusion_realmodel_deepseek.json")
    from reasoning.instinct_validation import _dp_auc
    _, ins = _dp_auc(insample_qw_fused(scores, labels), labels)
    _, loo = _dp_auc(loo_qw_fused(scores, labels), labels)
    assert loo <= ins + 1e-9


def test_bootstrap_ci_is_ordered_and_bounded():
    scores, labels = load_scores(RESULTS / "fusion_realmodel_deepseek.json")
    lo, hi = bootstrap_ci(scores["B2"], labels, metric="auc", trials=500, seed=1)
    assert 0.0 <= lo <= hi <= 1.0


def test_label_free_reflex_weaker_than_structural_verifier():
    ev = evaluate(RESULTS / "fusion_realmodel_deepseek.json")
    a = ev["rows"]["A"]
    b2 = ev["rows"]["B2"]
    assert a["auc"] < b2["auc"]            # self-consistency is the weak, honest one
    assert a["auc_ci"][0] <= 0.5           # A's CI still includes chance


def test_low_clean_model_flagged():
    ev = evaluate(RESULTS / "fusion_realmodel_llmhub-haiku.json")
    assert ev["n_clean"] < 5
    assert ev["low_clean_warning"] is True


def test_deepseek_equal_fusion_ci_excludes_chance():
    ev = evaluate(RESULTS / "fusion_realmodel_deepseek.json")
    assert ev["rows"]["fused_equal"]["auc_ci"][0] > 0.5


def test_cross_process_determinism():
    def once(seed_env: str) -> str:
        env = dict(os.environ, PYTHONHASHSEED=seed_env)
        return subprocess.run(
            [sys.executable, str(ROOT / "reasoning" / "instinct_validation.py"), "--self-test"],
            capture_output=True, text=True, env=env, check=True,
        ).stdout
    assert once("0") == once("123")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all instinct_validation tests passed")
