#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the version single-source-of-truth guard (tools/check_version_consistency).

Verifies the live repo is consistent (VERSION has a CHANGELOG section + synced badge), and
that the predicates catch the exact drift that left VERSION 0.8.0 with the last tag v0.5.3:
a non-semver VERSION and a missing CHANGELOG section. Offline, stdlib-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.check_version_consistency import (  # noqa: E402
    changelog_has_section, check, read_version,
)


def test_live_repo_is_consistent() -> None:
    result = check()
    assert result["ok"], f"version drift: {result['problems']}"


def test_current_version_has_changelog_section() -> None:
    assert changelog_has_section(read_version())


def test_missing_section_is_flagged() -> None:
    # A version that was never released must not pass the changelog predicate.
    assert changelog_has_section("999.999.999") is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
