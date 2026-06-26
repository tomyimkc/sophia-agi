#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for false-positive-regression diagnosis (eval self-diagnosis + standalone tool)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnose_fp_regression import regressions_from_report  # noqa: E402
from tools.eval_rlvr_adapter import false_positive_regressions  # noqa: E402


def _row(cid, label, reward, *, completion="", denied=False):
    return {"case_id": cid, "label": label, "work": cid + "-work", "reward": reward,
            "completion": completion, "detail": {"deniedOnTrueCase": denied}}


def test_flags_true_case_that_flipped_to_false_positive() -> None:
    base = [_row("t1", "true", 1.0), _row("t2", "true", 1.0), _row("f1", "false", 0.7)]
    # adapter regresses t2 (true case) to a denial false-positive; t1 stays good.
    adapter = [_row("t1", "true", 1.0), _row("t2", "true", -0.5, completion="No, X did not...", denied=True),
               _row("f1", "false", 0.7)]
    regs = false_positive_regressions(base, adapter)
    assert len(regs) == 1 and regs[0]["case_id"] == "t2"
    assert regs[0]["adapterDeniedOnTrueCase"] is True


def test_no_regression_when_integrity_held() -> None:
    base = [_row("t1", "true", 1.0), _row("f1", "false", 0.7)]
    adapter = [_row("t1", "true", 1.0), _row("f1", "false", 1.0)]
    assert false_positive_regressions(base, adapter) == []


def test_false_case_changes_are_not_fp_regressions() -> None:
    # A FALSE case getting worse is not a true-case false-positive (different metric).
    base = [_row("f1", "false", 0.7)]
    adapter = [_row("f1", "false", -1.0)]
    assert false_positive_regressions(base, adapter) == []


def test_regressions_from_report_prefers_precomputed() -> None:
    report = {"falsePositiveRegressions": [{"case_id": "t9"}], "rows": {"base": [], "adapter": []}}
    assert regressions_from_report(report) == [{"case_id": "t9"}]


def test_regressions_from_report_derives_from_rows() -> None:
    report = {
        "rows": {
            "base": [_row("t2", "true", 1.0)],
            "adapter": [_row("t2", "true", -0.5, denied=True)],
        }
    }
    regs = regressions_from_report(report)
    assert len(regs) == 1 and regs[0]["case_id"] == "t2"


def main() -> int:
    test_flags_true_case_that_flipped_to_false_positive()
    test_no_regression_when_integrity_held()
    test_false_case_changes_are_not_fp_regressions()
    test_regressions_from_report_prefers_precomputed()
    test_regressions_from_report_derives_from_rows()
    print("test_diagnose_fp_regression: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
