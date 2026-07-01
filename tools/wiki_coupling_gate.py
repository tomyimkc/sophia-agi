#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Load-bearing HONEST gate over OKF wiki edge-mining PROPOSALS.

This gate scores the PROPOSED evidence-edge graph (from okf.evidence_edges) and
the ignorance-as-a-node overlay (okf.gap_nodes) against PRE-REGISTERED floors in
agi-proof/edge-mining/coupling_floors.json. It reports on PROPOSALS ONLY — the
wiki is a generated artifact and this gate never reads or writes a wiki page's
truth, only measures the proposed overlay. It cannot and does not claim AGI.

Metrics gated:
  edge density                edges / page                        >= minEdgeDensity
  cross-theme coupling        frac. edges spanning two areas       >= minCrossThemeCoupling
  grounded-ignorance coverage frac. gap nodes linked to >=1 concept>= minGroundedIgnoranceCoverage
  precision proxy             frac. edges with >=2 signals         >= precisionProxyFloor

The precision proxy is the ANTI-GOODHART floor: count-only floors are trivially
gamed by emitting many single-signal edges, so the gate FAILS when the proxy is
below floor even if density clears. A gate that always passes is worse than none.

Exit 0 = PASS (all floors cleared). Exit 1 = FAIL (any floor missed). Exit 2 =
unreadable inputs. JSON receipt to stdout; human prose to stderr.

    python3 tools/wiki_coupling_gate.py
    python3 tools/wiki_coupling_gate.py --root wiki --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf import evidence_edges, gap_nodes  # noqa: E402

DEFAULT_ROOTS = [ROOT / "wiki"]
ATTRIBUTIONS_PATH = ROOT / "data" / "attributions.json"
LEDGER_PATH = ROOT / "agi-proof" / "failure-ledger.md"
MANIFEST_PATH = ROOT / "agi-proof" / "evidence-manifest.json"
FLOORS_PATH = ROOT / "agi-proof" / "edge-mining" / "coupling_floors.json"

_DEFAULT_FLOORS = {
    "minEdgeDensity": 0.5,
    "minCrossThemeCoupling": 0.15,
    "minGroundedIgnoranceCoverage": 0.5,
    "precisionProxyFloor": 0.35,
    "minEdgeScoreForProxy": 0.0,
    "minSignalsForPrecise": 2,
}


def load_floors(path: Path) -> dict:
    """Load pre-registered floors; fall back to conservative defaults if absent."""
    floors = dict(_DEFAULT_FLOORS)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return floors
        if isinstance(data, dict):
            floors.update(data.get("floors", {}))
    return floors


def evaluate(roots, floors: dict) -> dict:
    """Compute metrics + per-floor pass/fail. Pure (no file writes)."""
    paths = [Path(r) for r in roots if Path(r).exists()]
    pages = okf.load_pages(*paths)
    attributions = evidence_edges.load_attributions(ATTRIBUTIONS_PATH)
    edges = evidence_edges.mine_edges(pages, attributions=attributions)

    gaps = gap_nodes.load_gaps(LEDGER_PATH, MANIFEST_PATH)
    gap_nodes.link_gaps_to_concepts(gaps, pages)
    gap_cov = gap_nodes.coverage(gaps)

    metrics = evidence_edges.coupling_metrics(
        pages, edges,
        min_signals_for_precise=int(floors["minSignalsForPrecise"]),
        min_edge_score_for_proxy=float(floors["minEdgeScoreForProxy"]),
    )

    checks = {
        "edgeDensity": {
            "value": metrics["edgeDensity"],
            "floor": floors["minEdgeDensity"],
            "pass": metrics["edgeDensity"] >= floors["minEdgeDensity"],
        },
        "crossThemeCoupling": {
            "value": metrics["crossThemeCoupling"],
            "floor": floors["minCrossThemeCoupling"],
            "pass": metrics["crossThemeCoupling"] >= floors["minCrossThemeCoupling"],
        },
        "groundedIgnoranceCoverage": {
            "value": gap_cov["coverage"],
            "floor": floors["minGroundedIgnoranceCoverage"],
            "pass": gap_cov["coverage"] >= floors["minGroundedIgnoranceCoverage"],
        },
        "precisionProxy": {
            "value": metrics["precisionProxy"],
            "floor": floors["precisionProxyFloor"],
            "pass": metrics["precisionProxy"] >= floors["precisionProxyFloor"],
            "antiGoodhart": True,
        },
    }
    ok = all(c["pass"] for c in checks.values())
    failed = sorted(k for k, c in checks.items() if not c["pass"])
    return {
        "experimentId": "wiki-coupling-gate",
        "scope": "proposals-only; wiki is generated and is NOT modified",
        "canClaimAGI": False,
        "go": ok,
        "pass": ok,
        "pages": metrics["pages"],
        "edgeCount": metrics["edges"],
        "perKind": metrics["perKind"],
        "gapCount": gap_cov["gapCount"],
        "linkedGapCount": gap_cov["linkedGapCount"],
        "checks": checks,
        "failedChecks": failed,
        "floors": floors,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", dest="roots",
                    help="page root (repeatable); default: wiki/")
    ap.add_argument("--floors", default=str(FLOORS_PATH),
                    help="pre-registered floors JSON (default: agi-proof/edge-mining/coupling_floors.json)")
    ap.add_argument("--json", action="store_true", help="print full report to stdout")
    args = ap.parse_args(argv)

    try:
        floors = load_floors(Path(args.floors))
        roots = args.roots or [str(p) for p in DEFAULT_ROOTS]
        result = evaluate(roots, floors)
    except OSError as exc:  # unreadable inputs
        print(f"gate error: {exc}", file=sys.stderr)
        print(json.dumps({"experimentId": "wiki-coupling-gate", "error": str(exc),
                          "go": False, "canClaimAGI": False}, ensure_ascii=False))
        return 2

    verdict = "PASS" if result["pass"] else "FAIL"
    print(f"[wiki-coupling-gate] {verdict} — {result['edgeCount']} proposed edges over "
          f"{result['pages']} pages; failed: {result['failedChecks'] or '(none)'} "
          f"(proposals only; wiki NOT modified)", file=sys.stderr)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        receipt = {
            "experimentId": result["experimentId"],
            "pass": result["pass"],
            "go": result["go"],
            "canClaimAGI": result["canClaimAGI"],
            "failedChecks": result["failedChecks"],
            "checks": {k: {"value": v["value"], "floor": v["floor"], "pass": v["pass"]}
                       for k, v in result["checks"].items()},
        }
        print(json.dumps(receipt, ensure_ascii=False, indent=2))

    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
