#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Keep the README version badge in sync with the VERSION file.

The single source of truth for the version is the ``VERSION`` file. The README
carries a shields.io badge that drifts (it once said 0.7.27 while VERSION said
0.7.41). This tool derives the badge from VERSION so the drift cannot recur.

Usage:
  python tools/sync_readme_version.py            # rewrite the badge to match VERSION
  python tools/sync_readme_version.py --check     # exit 1 if the badge is stale (CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
README = ROOT / "README.md"

# Matches: ![Version](https://img.shields.io/badge/version-0.7.41-blue)
BADGE_RE = re.compile(r"(!\[Version\]\(https://img\.shields\.io/badge/version-)([^-)]+)(-[a-z]+\))")


def expected_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def current_badge_version(text: str) -> "str | None":
    m = BADGE_RE.search(text)
    return m.group(2) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync the README version badge with VERSION")
    parser.add_argument("--check", action="store_true", help="exit 1 if the badge is stale (no write)")
    args = parser.parse_args()

    version = expected_version()
    text = README.read_text(encoding="utf-8")
    found = current_badge_version(text)

    if found is None:
        print("ERROR: no version badge found in README.md", file=sys.stderr)
        return 2

    if found == version:
        print(f"OK: README badge matches VERSION ({version})")
        return 0

    if args.check:
        print(
            f"DRIFT: README badge is {found} but VERSION is {version}. "
            f"Run `python tools/sync_readme_version.py` to fix.",
            file=sys.stderr,
        )
        return 1

    new_text = BADGE_RE.sub(rf"\g<1>{version}\g<3>", text, count=1)
    README.write_text(new_text, encoding="utf-8")
    print(f"Updated README badge {found} -> {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
