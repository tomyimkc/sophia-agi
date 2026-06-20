#!/usr/bin/env python3
"""OKF wiki health metrics — for long-horizon autonomy + AGI-proof evidence.

One number per dimension of "is the knowledge base staying coherent on its own":
broken links, orphan pages, declared/structural contradictions, and provenance
violations. A long-horizon run logs these over time; non-degrading health under
unsupervised maintenance is a far stronger autonomy claim than "ran a loop."

    python tools/wiki_health.py            # print health JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import linker  # noqa: E402
from tools.lint_wiki_provenance import run_audit  # noqa: E402

WIKI_DIR = ROOT / "wiki"
DISPUTES_DIR = ROOT / "docs" / "04-Disputes"


def run() -> dict:
    roots = [p for p in (WIKI_DIR, DISPUTES_DIR) if p.exists()]
    report = linker.link_report(*roots)
    audit = run_audit()
    c = report["contradictions"]
    structural = len(c["selfMerges"]) + len(c["traditionMerges"]) + len(c["supersedeCycles"]) + len(c["confidenceLaundering"])
    metrics = {
        "pages": report["pages"],
        "backlinks": report["backlinkCount"],
        "brokenLinks": len(report["danglingLinks"]),
        "orphans": len(report["orphans"]),
        "schemaErrors": len(report["schemaErrors"]),
        "structuralContradictions": structural,
        "declaredContradictions": len(c["declaredContradictions"]),
        "provenanceViolations": len(audit["violations"]),
    }
    # "coherent" = no hard defects (orphans/declared-contradictions are informational)
    metrics["coherent"] = (
        metrics["brokenLinks"] == 0
        and metrics["schemaErrors"] == 0
        and metrics["structuralContradictions"] == 0
        and metrics["provenanceViolations"] == 0
    )
    return metrics


def main() -> int:
    metrics = run()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0 if metrics["coherent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
