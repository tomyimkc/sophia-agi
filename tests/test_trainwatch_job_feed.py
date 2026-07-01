#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for tools/trainwatch_job_feed.py — pure parser + guarded import.

Offline + deterministic. NO trainwatch installed here, which is the point: the module MUST import
(its trainwatch import is guarded behind follow()).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.trainwatch_job_feed import parse_progress  # noqa: E402


def test_hf_load_bar_pct_and_phase() -> None:
    p = parse_progress("Loading weights:  34%|##  | 60/179")
    assert p["phase"] == "loading"
    assert p["pct"] == round(100 * 60 / 179)  # 34
    assert p["metrics"]["n"] == 60 and p["metrics"]["total"] == 179


def test_generic_tqdm_eval() -> None:
    p = parse_progress("  37%|####  | 95/256 [00:10<00:20]")
    assert p["phase"] == "eval"
    assert p["pct"] == round(100 * 95 / 256)  # 37


def test_train_step_loss_lr() -> None:
    p = parse_progress("epoch 2/2 step 200/220 (90.9%) loss=0.5418 lr=1.93e-06")
    assert p["phase"] == "train"
    assert p["pct"] == round(100 * 200 / 220)  # 91
    assert p["metrics"]["loss"] == 0.5418
    assert p["metrics"]["lr"] == 1.93e-06
    assert p["metrics"]["step"] == 200 and p["metrics"]["total"] == 220


def test_train_step_no_lr() -> None:
    p = parse_progress("step 5/10 loss=1.0")
    assert p["pct"] == 50 and p["metrics"]["loss"] == 1.0
    assert "lr" not in p["metrics"]


def test_eval_val_loss() -> None:
    p = parse_progress("  [eval] step 150 val_loss=1.5012 train_loss=0.5373")
    assert p["phase"] == "eval"
    assert p["metrics"]["val_loss"] == 1.5012
    assert p["metrics"]["train_loss"] == 0.5373
    assert p["metrics"]["step"] == 150
    assert p["pct"] is None


def test_bench_step_label() -> None:
    p = parse_progress(">> STEP: A2 — Bench A")
    assert p["phase"] == "A2 — Bench A"
    assert p["pct"] is None
    p2 = parse_progress(">> STEP: B2 — Certify nvfp4")
    assert p2["phase"] == "B2 — Certify nvfp4"


def test_verdict_metric_extraction() -> None:
    p = parse_progress("VERDICT: PASS (mean_kl=0.045, top1=0.906)")
    assert p["phase"] == "done"
    assert p["pct"] == 100
    assert p["metrics"]["mean_kl"] == 0.045
    assert p["metrics"]["top1"] == 0.906


def test_verdict_scientific_and_top1_agreement() -> None:
    p = parse_progress("VERDICT: FAIL (mean_kl=1.2e-3, top1_agreement=0.9375)")
    assert p["metrics"]["mean_kl"] == 1.2e-3
    assert p["metrics"]["top1_agreement"] == 0.9375


def test_done_markers() -> None:
    for line in ("complete (exit 0)", "training finished (rc=0)", "saved adapter to ckpt",
                 "=== certify done", "training complete"):
        p = parse_progress(line)
        assert p is not None and p["phase"] == "done" and p["pct"] == 100, line


def test_fail_markers() -> None:
    for line in ("complete (exit 1)", "complete (exit 137)", "the run FAILED",
                 "Traceback (most recent call last):"):
        p = parse_progress(line)
        assert p is not None and p["phase"] == "failed", line


def test_unknown_line_is_none() -> None:
    for line in ("", "just some log text", "INFO loading config", "==========",
                 "hello world 12 34"):
        assert parse_progress(line) is None, line


def test_determinism() -> None:
    lines = [
        "Loading weights:  34%|##| 60/179",
        "epoch 2/2 step 200/220 (90.9%) loss=0.5418 lr=1.93e-06",
        "VERDICT: PASS (mean_kl=0.045, top1=0.906)",
        ">> STEP: A2 — Bench A",
        "nope",
    ]
    for line in lines:
        assert parse_progress(line) == parse_progress(line), line


def test_module_imports_without_trainwatch() -> None:
    # The point of the guarded import: importing the module must NOT require trainwatch.
    assert "trainwatch" not in sys.modules, "trainwatch must not be imported at module load"
    import tools.trainwatch_job_feed as mod
    assert hasattr(mod, "parse_progress") and hasattr(mod, "follow")
    # _try_trainwatch returns None (not installed here) WITHOUT raising
    assert mod._try_trainwatch() is None


def test_selftest_passes() -> None:
    import tools.trainwatch_job_feed as mod
    assert mod._selftest() == 0


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} trainwatch_job_feed tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
