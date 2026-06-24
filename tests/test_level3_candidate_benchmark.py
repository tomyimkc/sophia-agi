#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/run_level3_candidate_benchmark.py.

The central safety invariant: this tool may rehearse all three Level-3 lanes, but
must not emit artifacts that the real Level-3 gate can mistake for real evidence.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_level3_candidate_benchmark as bench  # noqa: E402
from tools.hidden_eval_protocol import validate_pack  # noqa: E402


def test_candidate_benchmark_runs_all_lanes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate"
        summary = bench.run_all(out)
        assert summary["candidateOnly"] is True
        assert summary["level3Evidence"] is False
        assert summary["allCandidateLanesOk"] is True
        lanes = {lane["lane"]: lane for lane in summary["lanes"]}
        assert lanes["hidden_full_comparison"]["ok"] is True
        assert lanes["distribution_shift"]["postTestCases"] == 10
        assert lanes["long_horizon_30m"]["ok"] is True
        assert lanes["long_horizon_30m"]["autonomy"]["substantive"] is False
        assert (out / "level3-candidate-summary.json").exists()


def test_candidate_hidden_artifact_cannot_satisfy_real_gate() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate"
        summary = bench.run_all(out, skip_long_horizon=True)
        hidden = json.loads(Path(summary["lanes"][0]["artifact"]).read_text(encoding="utf-8"))
        assert hidden["candidateOnly"] is True
        assert hidden["level3Evidence"] is False
        assert hidden["visibility"] == "revealed-after-eval"  # real gate requires private-hidden
        assert "candidate" in hidden["packId"]


def test_candidate_distribution_not_written_to_gate_result_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "candidate"
        summary = bench.run_all(out, skip_long_horizon=True)
        dist = Path([lane for lane in summary["lanes"] if lane["lane"] == "distribution_shift"][0]["artifact"])
        assert "level3-candidate-benchmark" not in str(out) or dist.exists()
        assert "learning-under-shift" not in str(dist)
        data = json.loads(dist.read_text(encoding="utf-8"))
        assert data["candidateOnly"] is True
        assert data["level3Evidence"] is False


def test_real_scaffold_is_gitignored_shape_and_structurally_valid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        private_root = Path(tmp) / "private"
        scaffold = bench.emit_real_scaffold(private_root, "2099-01-02")
        assert scaffold["evidence"] is False
        assert len(scaffold["written"]) == 8
        hidden_pack = json.loads((private_root / "hidden-evals" / "level3-2099-01-02" / "PACK.json").read_text(encoding="utf-8"))
        assert hidden_pack["visibility"] == "private-hidden"
        assert validate_pack(hidden_pack) == []

        shift_spec = json.loads((private_root / "shift" / "level3-shift-spec-2099-01-02.json").read_text(encoding="utf-8"))
        assert len(shift_spec["preTestPack"]["cases"]) == 10
        assert len(shift_spec["postTestPack"]["cases"]) == 10
        assert validate_pack(shift_spec["preTestPack"]) == []
        assert validate_pack(shift_spec["postTestPack"]) == []
        assert validate_pack(shift_spec["oldBenchmarkPack"]) == []

        long_spec = json.loads((private_root / "long-horizon" / "30min-2099-01-02.json").read_text(encoding="utf-8"))
        assert long_spec["runId"] == "level3-30min-2099-01-02"
        assert "_instructions" in long_spec


def main() -> int:
    test_candidate_benchmark_runs_all_lanes()
    test_candidate_hidden_artifact_cannot_satisfy_real_gate()
    test_candidate_distribution_not_written_to_gate_result_path()
    test_real_scaffold_is_gitignored_shape_and_structurally_valid()
    print("test_level3_candidate_benchmark: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
