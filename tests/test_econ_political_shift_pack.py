#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for Economics & Political Economy shift scaffold."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_econ_political_shift_pack as econ  # noqa: E402
from tools.hidden_eval_protocol import validate_pack  # noqa: E402


def test_econ_spec_is_multi_case_and_valid() -> None:
    spec = econ.econ_spec("2099-01-02")
    assert spec["domainLabel"].startswith("Economics")
    assert len(spec["learningRecords"]) == 12
    assert len(spec["preTestPack"]["cases"]) == 12
    assert len(spec["postTestPack"]["cases"]) == 12
    assert validate_pack(spec["preTestPack"]) == []
    assert validate_pack(spec["postTestPack"]) == []
    assert validate_pack(spec["oldBenchmarkPack"]) == []


def test_candidate_report_has_accept_and_hold() -> None:
    report = econ.build_candidate_report("2099-01-02")
    assert report["candidateOnly"] is True
    assert report["level3Evidence"] is False
    assert report["summary"]["accepted"] >= 2
    assert report["summary"]["held"] >= 1  # unsupported GDP claim must not pass


def test_cli_writes_to_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        private = Path(tmp) / "shift.json"
        cand = Path(tmp) / "candidate.json"
        assert econ.main.__call__ is not None
        econ.write_json(private, econ.econ_spec("2099-01-02"))
        econ.write_json(cand, econ.build_candidate_report("2099-01-02"))
        assert json.loads(private.read_text())["experimentId"].startswith("econ-political")
        assert json.loads(cand.read_text())["candidateOnly"] is True


def main() -> int:
    test_econ_spec_is_multi_case_and_valid()
    test_candidate_report_has_accept_and_hold()
    test_cli_writes_to_paths()
    print("test_econ_political_shift_pack: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
