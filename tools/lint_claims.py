#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Claims linter — fail the build if public copy makes a capability/safety overclaim.

The README and marketing must never exceed what the failure ledger supports. This
deterministic, offline linter scans public-facing prose for forbidden superlatives
and unqualified safety claims. A line may opt out by including the marker
``claim-ok`` (use sparingly, only when the surrounding text already qualifies it).

Run:  python tools/lint_claims.py
Exit: 0 = clean, 1 = violations found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files / globs whose prose is public-facing marketing or positioning.
SCAN: list[str] = [
    "README.md",
    "VISION.md",
    "CONTRIBUTING.md",
    "GOOD_FIRST_ISSUES.md",
    "SPONSORS.md",
    "agi-proof/README.md",
    "docs/00-Index/Home.md",
]
SCAN_GLOBS: list[str] = ["docs/07-Growth/**/*.md", "docs/07-Growth/**/*.txt"]

# (regex, why) — capability/safety overclaims that the ledger does not support.
FORBIDDEN: list[tuple[str, str]] = [
    (r"\bsafe to ship\b", "implies a guarantee; the gate is a filter (23.6% residual)"),
    (r"\btrust in production without\b", "implies no oversight needed"),
    (r"\bwithout constant oversight\b", "implies autonomy the evidence does not support"),
    (r"\bmakes ai safe\b", "unqualified safety claim"),
    (r"\b(the\s+)?first\s+.{0,12}\bagi\b", "AGI primacy claim"),
    (r"\bproven agi\b|\bis agi\b", "AGI capability claim"),
    (r"\bbirth(ing)?\s+the\s+first\b", "AGI primacy / hype"),
    (r"\bthe only open project\b", "unfalsifiable superiority claim"),
    (r"\b100%\s+on\s+all\b", "first-party benchmark stated as universal result"),
    (r"\bworld'?s first\b", "primacy claim"),
    (r"\bbreakthrough\b", "hype term without a cited result"),
    (r"\bproves alignment\b", "unformalizable alignment overclaim"),
    (r"\bproves safe self-improvement\b", "unqualified safe-self-improvement claim"),
    (r"\bgödel machine\b|\bgodel machine\b", "misleading Gödel-machine framing"),
    (r"\bproves it is trustworthy\b", "unqualified trustworthiness proof claim"),
]

ALLOW_MARKER = "claim-ok"


def _files() -> list[Path]:
    out: list[Path] = []
    for rel in SCAN:
        p = ROOT / rel
        if p.exists():
            out.append(p)
    for g in SCAN_GLOBS:
        out.extend(sorted(ROOT.glob(g)))
    return out


def main() -> int:
    violations: list[str] = []
    for path in _files():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if ALLOW_MARKER in line:
                continue
            low = line.lower()
            for pat, why in FORBIDDEN:
                if re.search(pat, low):
                    rel = path.relative_to(ROOT)
                    violations.append(f"{rel}:{i}: «{line.strip()[:90]}» — {why}")

    if violations:
        print("CLAIMS LINTER: FAIL — overclaims found (fix the copy or add a qualifier):\n")
        for v in violations:
            print("  " + v)
        print(f"\n{len(violations)} violation(s). The README must not exceed agi-proof/failure-ledger.md.")
        print("If a line is genuinely qualified in context, append the marker 'claim-ok'.")
        return 1
    print(f"CLAIMS LINTER: OK — scanned {len(_files())} file(s), no overclaims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
