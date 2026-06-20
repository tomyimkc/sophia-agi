#!/usr/bin/env python3
"""Provenance-faithfulness lint over the whole OKF wiki — the falsifier.

Runs agent.verifiers.provenance_faithful over every wiki + dispute page body. A
single forbidden attribution (an answer or page crossing a doNotAttributeTo edge)
is a pre-registered falsification event for Sophia's source-discipline claim, so
this exits non-zero and writes an audit report. Designed to scale as the wiki
grows autonomously (the librarian must never industrialise a lineage merge).

    python tools/lint_wiki_provenance.py            # exit 1 on any violation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verifiers  # noqa: E402
from okf import frontmatter  # noqa: E402

WIKI_DIR = ROOT / "wiki"
DISPUTES_DIR = ROOT / "docs" / "04-Disputes"
AUDIT_PATH = ROOT / "eval" / "provenance_audit.json"


def run_audit() -> dict:
    verifier = verifiers.provenance_faithful()
    pages = 0
    violations: list[dict] = []
    for root in (WIKI_DIR, DISPUTES_DIR):
        if not root.exists():
            continue
        for md in sorted(root.rglob("*.md")):
            pages += 1
            body = frontmatter.strip(md.read_text(encoding="utf-8"))
            result = verifier(body, None, {})
            if not result["passed"]:
                violations.append({"page": str(md.relative_to(ROOT)), "reasons": result["reasons"]})
    return {"ok": not violations, "pagesScanned": pages, "violations": violations}


def main() -> int:
    result = run_audit()
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not result["ok"]:
        print(f"Provenance lint FAILED — {len(result['violations'])} forbidden attribution(s):")
        for item in result["violations"]:
            print(f"  - {item['page']}: {'; '.join(item['reasons'])}")
        return 1
    print(f"Provenance lint OK: {result['pagesScanned']} page(s) scanned, 0 forbidden attributions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
