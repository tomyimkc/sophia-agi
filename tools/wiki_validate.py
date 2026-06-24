#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the OKF wiki: frontmatter schema, link integrity, contradictions, drift.

Mirrors tools/validate_attribution.py (run_validation() -> dict, main() -> int) so
it slots into CI right after the attribution check. A failure means the wiki is no
longer a clean provenance graph: a schema error, a dangling [[wikilink]], a
lineage-merge/supersede cycle, or frontmatter that drifted from data/*.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import linker  # noqa: E402
from tools import wiki_sync  # noqa: E402

WIKI_DIR = ROOT / "wiki"
DISPUTES_DIR = ROOT / "docs" / "04-Disputes"


def run_validation() -> dict:
    roots = [p for p in (WIKI_DIR, DISPUTES_DIR) if p.exists()]
    if not roots:
        return {"ok": False, "errors": [f"no wiki found (looked in {WIKI_DIR}, {DISPUTES_DIR})"]}

    report = linker.link_report(*roots)
    drift = wiki_sync.check()

    errors: list[str] = []
    for item in report["schemaErrors"]:
        errors.append(f"schema {item['page']}: {'; '.join(item['errors'])}")
    for d in report["danglingLinks"]:
        errors.append(f"dangling link {d['page']} -> [[{d['target']}]]")
    contradictions = report["contradictions"]
    for sm in contradictions["selfMerges"]:
        errors.append(f"lineage-merge: {sm['page']} attributes to do-not-attribute author '{sm['author']}'")
    for tm in contradictions["traditionMerges"]:
        errors.append(f"tradition-merge: {tm['page']} links {tm['otherTradition']} (do-not-merge)")
    for cyc in contradictions["supersedeCycles"]:
        errors.append(f"supersede cycle: {' -> '.join(cyc)}")
    for cl in contradictions["confidenceLaundering"]:
        errors.append(f"confidence-laundering: {cl['page']} claims '{cl['claims']}' above provenance")
    for miss in drift["missing"]:
        errors.append(f"missing generated page: {miss} (run `python tools/wiki_sync.py emit`)")
    for dr in drift["drift"]:
        errors.append(f"drift {dr['page']} [{dr['key']}]: wiki={dr['wiki']!r} != data={dr['data']!r}")

    return {
        "ok": not errors,
        "pages": report["pages"],
        "backlinks": report["backlinkCount"],
        "orphans": len(report["orphans"]),
        "declaredContradictions": len(contradictions["declaredContradictions"]),
        "errors": errors,
    }


def main() -> int:
    result = run_validation()
    if not result["ok"]:
        print("Wiki validation FAILED:")
        for err in result["errors"]:
            print(f"  - {err}")
        return 1
    print(
        f"Wiki validation OK: {result['pages']} page(s), {result['backlinks']} backlink(s), "
        f"{result['orphans']} orphan(s), {result['declaredContradictions']} declared contradiction(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
