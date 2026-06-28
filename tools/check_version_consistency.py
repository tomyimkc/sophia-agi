#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Guard the version single-source-of-truth so VERSION can't drift from its evidence.

The ``VERSION`` file is the single source of truth. This check fails (exit 1) unless:
  1. ``VERSION`` is valid semver ``X.Y.Z``;
  2. ``CHANGELOG.md`` has a ``## [X.Y.Z]`` section for it (you can't bump the version
     without release notes — and the auto-release workflow needs that section);
  3. the README shields.io version badge matches ``VERSION``
     (delegated to ``tools/sync_readme_version.py --check``).

Wiring this into CI is what stops the failure that left the repo at VERSION 0.8.0 while the
last git tag was v0.5.3: a bump with no notes and no tag. Pair it with the release workflow
(`.github/workflows/release.yml`), which tags + publishes whenever VERSION changes on main.

  python tools/check_version_consistency.py            # verify, exit 1 on drift
  python tools/check_version_consistency.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
CHANGELOG = ROOT / "CHANGELOG.md"

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def read_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def changelog_has_section(version: str) -> bool:
    text = CHANGELOG.read_text(encoding="utf-8")
    return re.search(rf"^## \[{re.escape(version)}\]", text, re.MULTILINE) is not None


def readme_badge_ok() -> "tuple[bool, str]":
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "sync_readme_version.py"), "--check"],
                          capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def check() -> dict:
    version = read_version()
    problems: list[str] = []
    if not _SEMVER.match(version):
        problems.append(f"VERSION '{version}' is not semver X.Y.Z")
    if not changelog_has_section(version):
        problems.append(f"CHANGELOG.md has no '## [{version}]' section — add release notes "
                        f"(or promote [Unreleased]) before bumping VERSION")
    badge_ok, badge_msg = readme_badge_ok()
    if not badge_ok:
        problems.append(f"README version badge out of sync: {badge_msg} "
                        f"(fix: python tools/sync_readme_version.py)")
    return {"version": version, "ok": not problems, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check VERSION / CHANGELOG / README consistency")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    result = check()
    if args.json:
        print(json.dumps(result, indent=2))
    elif result["ok"]:
        print(f"version-consistency OK — VERSION {result['version']} has a CHANGELOG section "
              f"and the README badge matches.")
    else:
        print("version-consistency FAILED:", file=sys.stderr)
        for p in result["problems"]:
            print(f"  - {p}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
