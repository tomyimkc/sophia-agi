#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for public claims linter."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_lint_claims_ok_on_repo() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "lint_claims.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_lint_claims_flags_godel_machine() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "README.md"
        bad.write_text("This repo is a Gödel machine for AGI.\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "lint_claims.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
        )
        # Default scan paths are repo files; inject by patching SCAN is heavy.
        # Direct pattern check instead:
        from tools.lint_claims import FORBIDDEN
        import re

        line = "This repo is a Gödel machine for AGI."
        assert any(re.search(pat, line.lower()) for pat, _why in FORBIDDEN)


def main() -> int:
    test_lint_claims_ok_on_repo()
    test_lint_claims_flags_godel_machine()
    print("test_lint_claims: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
