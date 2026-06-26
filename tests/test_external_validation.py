#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline, deterministic tests for the third-party SEIB uplift reproducer.

No model required: these exercise the verdict logic, the paired bootstrap CI, the
pack-hash pin, and the decontam fail-closed behavior with synthetic inputs, plus one
guard that the committed pack still matches the pre-registration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_external_validation import (  # noqa: E402
    DEFAULT_PREREG,
    paired_delta_ci,
    recompute_verdict,
    sha256_file,
)

PREREG = json.loads((ROOT / DEFAULT_PREREG).read_text(encoding="utf-8"))
GOOD_SHA = PREREG["evalPack"]["sha256"]


def _report(*, fa=0.0, fab=0.0, fp=0.02, acc_full=0.96, acc_raw=0.80, n=100, runs=3, real=True):
    return {
        "nCases": n, "runs": runs, "realModelRun": real, "ok": True,
        "byCondition": {
            "raw": {"provenanceAccuracy": acc_raw, "falseAttributionRate": 0.3,
                    "fabricationRateOnContested": 0.2, "falsePositiveCost": 0.0, "sourceCitationRate": 0.0},
            "sophia_full": {"provenanceAccuracy": acc_full, "falseAttributionRate": fa,
                            "fabricationRateOnContested": fab, "falsePositiveCost": fp, "sourceCitationRate": 0.3},
        },
        "deltas": {"raw_to_full_accuracy_delta": round(acc_full - acc_raw, 4)},
    }


def _clean_decontam():
    return {"evaluable": True, "overlapCount": 0, "nTrainRows": 1336}


def _ci_pos():
    return {"point": 0.16, "ci95": [0.05, 0.27], "nCases": 100}


def test_clean_validating_run_passes() -> None:
    v = recompute_verdict(_report(), PREREG, ci=_ci_pos(), decontam=_clean_decontam(),
                          pack_sha=GOOD_SHA, allow_mock=False)
    assert v["verdict"] == "PASS" and v["validating"] is True, v["checks"]


def test_false_attribution_fails() -> None:
    v = recompute_verdict(_report(fa=0.02), PREREG, ci=_ci_pos(), decontam=_clean_decontam(),
                          pack_sha=GOOD_SHA, allow_mock=False)
    assert v["verdict"] == "FAIL"
    assert any(c["check"] == "falseAttributionRate" and not c["pass"] for c in v["checks"])


def test_negative_or_zero_delta_fails() -> None:
    # raw matches full -> delta 0 and CI includes 0: the core falsification rule.
    v = recompute_verdict(_report(acc_full=0.80), PREREG, ci={"ci95": [-0.05, 0.05]},
                          decontam=_clean_decontam(), pack_sha=GOOD_SHA, allow_mock=False)
    assert v["verdict"] == "FAIL"
    assert any(c["check"] == "accuracyDelta" and not c["pass"] for c in v["checks"])


def test_ci_including_zero_fails_even_if_point_positive() -> None:
    v = recompute_verdict(_report(), PREREG, ci={"ci95": [-0.01, 0.3]}, decontam=_clean_decontam(),
                          pack_sha=GOOD_SHA, allow_mock=False)
    assert any(c["check"] == "accuracyDeltaCIExcludesZero" and not c["pass"] for c in v["checks"])
    assert v["verdict"] == "FAIL"


def test_high_false_positive_cost_fails() -> None:
    v = recompute_verdict(_report(fp=0.25), PREREG, ci=_ci_pos(), decontam=_clean_decontam(),
                          pack_sha=GOOD_SHA, allow_mock=False)
    assert v["verdict"] == "FAIL"


def test_decontam_overlap_fails() -> None:
    dirty = {"evaluable": True, "overlapCount": 3, "nTrainRows": 1336}
    v = recompute_verdict(_report(), PREREG, ci=_ci_pos(), decontam=dirty, pack_sha=GOOD_SHA, allow_mock=False)
    assert v["verdict"] == "FAIL"
    assert any(c["check"] == "decontamination" and not c["pass"] for c in v["checks"])


def test_empty_decontam_audit_is_not_clean() -> None:
    empty = {"evaluable": False, "overlapCount": 0, "nTrainRows": 0}
    v = recompute_verdict(_report(), PREREG, ci=_ci_pos(), decontam=empty, pack_sha=GOOD_SHA, allow_mock=False)
    assert v["verdict"] == "FAIL"  # 0 rows audited must not pass as "clean"


def test_pack_sha_mismatch_fails_and_blocks_validation() -> None:
    v = recompute_verdict(_report(), PREREG, ci=_ci_pos(), decontam=_clean_decontam(),
                          pack_sha="deadbeef", allow_mock=False)
    assert v["verdict"] == "FAIL" and v["validating"] is False


def test_mock_passes_plumbing_but_never_validates() -> None:
    v = recompute_verdict(_report(real=False, runs=1), PREREG, ci=_ci_pos(), decontam=_clean_decontam(),
                          pack_sha=GOOD_SHA, allow_mock=True)
    assert v["verdict"] == "PASS" and v["validating"] is False


def test_single_run_real_does_not_validate() -> None:
    v = recompute_verdict(_report(runs=1), PREREG, ci=_ci_pos(), decontam=_clean_decontam(),
                          pack_sha=GOOD_SHA, allow_mock=False)
    assert v["validating"] is False
    assert any(c["check"] == "minRuns" and not c["pass"] for c in v["checks"])


def test_paired_delta_ci_detects_uplift() -> None:
    rows = []
    for i in range(40):
        rows.append({"id": f"c{i}", "condition": "raw", "score": {"correct": i % 4 == 0}})       # 25%
        rows.append({"id": f"c{i}", "condition": "sophia_full", "score": {"correct": i % 4 != 3}})  # 75%
    ci = paired_delta_ci(rows, bootstrap=500)
    assert ci is not None and ci["point"] > 0 and ci["ci95"][0] > 0


def test_committed_pack_matches_preregistration() -> None:
    """Guard against silent pack drift — like SEIB's own n==100 assertion."""
    assert sha256_file(ROOT / PREREG["evalPack"]["path"]) == GOOD_SHA


def main() -> int:
    test_clean_validating_run_passes()
    test_false_attribution_fails()
    test_negative_or_zero_delta_fails()
    test_ci_including_zero_fails_even_if_point_positive()
    test_high_false_positive_cost_fails()
    test_decontam_overlap_fails()
    test_empty_decontam_audit_is_not_clean()
    test_pack_sha_mismatch_fails_and_blocks_validation()
    test_mock_passes_plumbing_but_never_validates()
    test_single_run_real_does_not_validate()
    test_paired_delta_ci_detects_uplift()
    test_committed_pack_matches_preregistration()
    print("test_external_validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
