#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mine typed evidence edges over the OKF wiki and write a PROPOSAL report.

Runs okf.evidence_edges over wiki/ (plus optional roots), builds ignorance-as-a-
node gaps from the failure ledger + evidence manifest, and writes a machine-
readable proposal to agi-proof/edge-mining/proposed-edges.json. It is strictly a
PROPOSAL engine: it NEVER modifies any wiki page (wiki/**/*.md is generated), it
only emits a report. Exit 0 on success; prints a JSON receipt to stdout and human
prose to stderr.

    python3 tools/mine_evidence_edges.py
    python3 tools/mine_evidence_edges.py --root wiki --min-score 0.3 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import evidence_edges, gap_nodes, load_pages  # noqa: E402

DEFAULT_ROOTS = [ROOT / "wiki"]
DEFAULT_OUT = ROOT / "agi-proof" / "edge-mining" / "proposed-edges.json"
ATTRIBUTIONS_PATH = ROOT / "data" / "attributions.json"
LEDGER_PATH = ROOT / "agi-proof" / "failure-ledger.md"
MANIFEST_PATH = ROOT / "agi-proof" / "evidence-manifest.json"
FLOORS_PATH = ROOT / "agi-proof" / "edge-mining" / "coupling_floors.json"


def _load_floors(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data.get("floors", {}) if isinstance(data, dict) else {}


def build_report(roots, *, min_score: float) -> dict:
    """Build the full proposal report (no file I/O for the report itself)."""
    paths = [Path(r) for r in roots if Path(r).exists()]
    pages = load_pages(*paths)
    attributions = evidence_edges.load_attributions(ATTRIBUTIONS_PATH)

    edges = evidence_edges.mine_edges(pages, attributions=attributions, min_score=min_score)

    gaps = gap_nodes.load_gaps(LEDGER_PATH, MANIFEST_PATH)
    gap_nodes.link_gaps_to_concepts(gaps, pages)
    gap_cov = gap_nodes.coverage(gaps)

    floors = _load_floors(FLOORS_PATH)
    metrics = evidence_edges.coupling_metrics(
        pages, edges,
        min_signals_for_precise=int(floors.get("minSignalsForPrecise", 2)),
        min_edge_score_for_proxy=float(floors.get("minEdgeScoreForProxy", 0.0)),
    )
    metrics["groundedIgnoranceCoverage"] = gap_cov["coverage"]

    return {
        "experimentId": "wiki-edge-mining",
        "status": "proposal_only",
        "canClaimAGI": False,
        "note": "Proposals over generated wiki/ pages. NO wiki file is modified.",
        "minScore": min_score,
        "pages": metrics["pages"],
        "edgeCount": metrics["edges"],
        "perKind": metrics["perKind"],
        "metrics": metrics,
        "gaps": gaps,
        "gapCoverage": gap_cov,
        "edges": edges,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", dest="roots",
                    help="page root (repeatable); default: wiki/")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output proposal path (default: agi-proof/edge-mining/proposed-edges.json)")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="drop proposed edges below this score (default 0.0)")
    ap.add_argument("--no-write", action="store_true",
                    help="compute + print receipt but do not write the proposal file")
    ap.add_argument("--json", action="store_true", help="print full report to stdout")
    args = ap.parse_args(argv)

    roots = args.roots or [str(p) for p in DEFAULT_ROOTS]
    report = build_report(roots, min_score=args.min_score)

    out_path = Path(args.out)
    if not args.no_write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "experimentId": report["experimentId"],
        "status": report["status"],
        "canClaimAGI": report["canClaimAGI"],
        "pages": report["pages"],
        "edgeCount": report["edgeCount"],
        "perKind": report["perKind"],
        "crossThemeCoupling": report["metrics"]["crossThemeCoupling"],
        "precisionProxy": report["metrics"]["precisionProxy"],
        "groundedIgnoranceCoverage": report["gapCoverage"]["coverage"],
        "gapCount": report["gapCoverage"]["gapCount"],
        "wikiModified": False,
        "output": None if args.no_write else str(out_path),
    }

    print(f"(mined {report['edgeCount']} proposed edges over {report['pages']} pages; "
          f"NO wiki file modified)", file=sys.stderr)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
