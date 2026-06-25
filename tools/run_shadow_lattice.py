#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the Shadow OKF bulk-boundary lattice pilot (candidate infrastructure).

Explores hypothetical derivesFrom / cross-tradition links in bulk, then projects
only gate-clean nodes toward the boundary. Bulk output is never shipped raw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_store  # noqa: E402
from okf.bulk_graph import BulkGraph  # noqa: E402
from okf.graph import build as build_graph  # noqa: E402
from okf.projection import project_to_boundary  # noqa: E402
from okf.promotion_loop import submit_projection_candidates  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "shadow-okf-lattice" / "shadow-lattice.public-report.json"


def _load_dnm() -> dict:
    path = ROOT / "data" / "traditions.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v.get("doNotMergeWith", []) for k, v in raw.items()}


def build_demo_report(*, skip_provenance: bool = False, skip_conscience: bool = False) -> dict:
    """Deterministic offline demo: one clean bulk node + one tradition-violating hypothesis."""
    boundary = build_graph(wiki_store.belief_graph_pages())
    bulk = BulkGraph(boundary=boundary, relax_tradition=True)

    bulk.add_node(
        "bulk_hypothesis_clean",
        meta={
            "id": "bulk_hypothesis_clean",
            "pageType": "concept",
            "tradition": "daoist",
            "authorConfidence": "disputed",
            "derivesFrom": ["dao_de_jing"],
        },
        body=(
            "Hypothesis (bulk only, candidateOnly): the Dao De Jing may be read philosophically "
            "and religiously without collapsing the two receptions. This is a tentative "
            "cross-reading, not a settled attribution. "
            "Sophia is an AGI-candidate verifier-gated epistemic framework; this bulk note is "
            "candidate infrastructure only. 中文：試探性閱讀。"
        ),
    )

    bulk.add_node(
        "bulk_tradition_explore",
        meta={
            "id": "bulk_tradition_explore",
            "pageType": "concept",
            "tradition": "daoist",
            "authorConfidence": "disputed",
            "allowTraditionExploration": True,
            "links": ["great_learning"],
        },
        body=(
            "Bulk exploration (candidateOnly): a hypothetical link between Daoist and Confucian "
            "ritual concepts for counterfactual study only — not boundary truth. "
            "Sophia is an AGI-candidate verifier-gated epistemic framework; this bulk note is "
            "candidate infrastructure only."
        ),
    )
    bulk.add_hypothesis("bulk_tradition_explore", "links", "great_learning", note="cross-tradition probe")

    bulk.add_node(
        "bulk_lineage_trap",
        meta={
            "id": "bulk_lineage_trap",
            "pageType": "text",
            "attributedAuthor": "confucius",
            "doNotAttributeTo": ["confucius"],
            "authorConfidence": "legendary",
        },
        body="Trap: forbidden self-attribution should abstain on projection.",
    )

    projection = project_to_boundary(
        bulk,
        dnm_by_tradition=_load_dnm(),
        skip_provenance=skip_provenance,
        skip_conscience=skip_conscience,
    )
    promotion = submit_projection_candidates(projection, source="shadow_okf_lattice_demo")

    return {
        "schema": "sophia.shadow_okf_lattice.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "claimBoundary": (
            "Shadow OKF bulk-boundary lattice pilot. Bulk states are quarantined; "
            "only projection-passing nodes are promotion candidates. Not AGI."
        ),
        "bulkAudit": bulk.audit_entry(),
        "projection": projection.to_dict(),
        "promotion": promotion,
        "invariants": {
            "bulk_never_committed_raw": True,
            "lineage_trap_abstained": any(
                a.get("nodeId") == "bulk_lineage_trap" for a in projection.abstained
            ),
            "at_least_one_promoted": len(projection.promoted) >= 1,
        },
        "ok": len(projection.promoted) >= 1
        and any(a.get("nodeId") == "bulk_lineage_trap" for a in projection.abstained),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Shadow OKF bulk-boundary lattice pilot")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-provenance", action="store_true", help="offline CI: skip NLP provenance gate")
    ap.add_argument("--skip-conscience", action="store_true", help="offline CI: skip conscience gate")
    args = ap.parse_args()

    report = build_demo_report(skip_provenance=args.skip_provenance, skip_conscience=args.skip_conscience)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "out": str(out), "promoted": report["projection"]["promotedCount"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
