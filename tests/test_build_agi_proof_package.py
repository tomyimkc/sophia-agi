#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the AGI proof manifest builder check mode."""

from __future__ import annotations

import contextlib
import io
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_agi_proof_package as builder  # noqa: E402


def test_check_manifest_preserves_generated_and_does_not_write(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "evidence-manifest.json"
    manifest = {"generated": "2026-06-25T12:00:00", "value": 1}
    output.write_text(builder.manifest_text(manifest), encoding="utf-8")

    def fake_build_manifest(*, generated=None):
        return {"generated": generated, "value": 1}

    before = output.read_text(encoding="utf-8")
    with patch.object(builder, "build_manifest", fake_build_manifest):
        assert builder.check_manifest(output) == 0
    assert output.read_text(encoding="utf-8") == before


def test_check_manifest_reports_drift_without_writing(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "evidence-manifest.json"
    stale_manifest = {"generated": "2026-06-25T12:00:00", "value": 0}
    output.write_text(builder.manifest_text(stale_manifest), encoding="utf-8")

    def fake_build_manifest(*, generated=None):
        return {"generated": generated, "value": 1}

    before = output.read_text(encoding="utf-8")
    stderr = io.StringIO()
    with patch.object(builder, "build_manifest", fake_build_manifest), contextlib.redirect_stderr(stderr):
        assert builder.check_manifest(output) == 1
    assert output.read_text(encoding="utf-8") == before
    assert "DRIFT" in stderr.getvalue()


def test_write_manifest_keeps_default_write_behavior(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "evidence-manifest.json"

    def fake_build_manifest(*, generated=None):
        return {"generated": generated or "now", "value": 1}

    with patch.object(builder, "build_manifest", fake_build_manifest):
        assert builder.write_manifest(output) == output
    assert output.read_text(encoding="utf-8") == builder.manifest_text({"generated": "now", "value": 1})


def test_check_manifest_fails_closed_on_non_dict_manifest(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = tmp_path / "evidence-manifest.json"
    # Valid JSON that is not an object (a list) and outright malformed JSON must both
    # report drift (return 1), never raise — fail-closed.
    for bad in ("[1, 2, 3]", "{not valid json"):
        output.write_text(bad, encoding="utf-8")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            assert builder.check_manifest(output) == 1
        assert "DRIFT" in stderr.getvalue()
        assert str(output) in stderr.getvalue()  # message names the actual path


def main() -> int:
    with TemporaryDirectory() as td:
        test_check_manifest_preserves_generated_and_does_not_write(Path(td) / "clean")
        test_check_manifest_reports_drift_without_writing(Path(td) / "stale")
        test_write_manifest_keeps_default_write_behavior(Path(td) / "write")
        test_check_manifest_fails_closed_on_non_dict_manifest(Path(td) / "malformed")
    print("test_build_agi_proof_package: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
