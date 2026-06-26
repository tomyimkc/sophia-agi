#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Close the self-correction loop: knowledge-gap ledger → reviewable draft stubs.

Reads the gap ledger written by grounded/verified search (`agent.knowledge_gap_log`), plans
which *missing-topic* gaps become provenance-skeleton draft pages (`agent.gap_ingest`), and —
only when ``--write`` is passed — materializes them into the quarantined draft tier
(`wiki/drafts/`, gated). **Dry-run by default**: nothing is written unless you ask.

Stubs carry NO claims (authorConfidence `none_extant`, needsReview), so they are fail-closed —
the grounded router auto-abstains on them. The loop makes an unknown a *known unknown*, ready
for a sourced fill; it never fabricates provenance.

  python tools/close_gap_loop.py --gaps PATH            # dry-run: show what would be created
  python tools/close_gap_loop.py --gaps PATH --write     # materialize draft stubs (gated)
  python tools/close_gap_loop.py --gaps PATH --json      # machine-readable report
  python tools/close_gap_loop.py --gaps PATH --min-hits 2  # only topics queried >= 2x
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GAPS = ROOT / "agent" / "memory" / "knowledge_gaps.jsonl"


def run(gaps_path: Path, *, write: bool = False, min_hits: int = 1) -> dict:
    from agent.gap_ingest import live_page_ids, materialize, plan_ingestion
    from agent.knowledge_gap_log import load_gaps

    gaps = load_gaps(gaps_path)
    plan = plan_ingestion(gaps, existing_ids=live_page_ids(), min_hits=min_hits)
    report = materialize(plan, write=write)
    report["gapsRead"] = len(gaps)
    report["gapsPath"] = str(gaps_path)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Close the gap loop: gaps -> draft stubs")
    ap.add_argument("--gaps", type=Path, default=DEFAULT_GAPS, help="knowledge-gap ledger (jsonl)")
    ap.add_argument("--write", action="store_true", help="actually write draft stubs (default: dry-run)")
    ap.add_argument("--min-hits", type=int, default=1, help="only materialize topics queried >= N times")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if not args.gaps.exists():
        print(f"no gap ledger at {args.gaps} (nothing logged yet)")
        return 0

    report = run(args.gaps, write=args.write, min_hits=args.min_hits)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"Gaps read: {report['gapsRead']}  (from {Path(report['gapsPath']).name})")
    if report["wrote"]:
        print(f"  created {len(report['created'])} draft stub(s) in {report['tier']} tier:")
        for c in report["created"]:
            print(f"    + {c['id']}  ({c['hits']}× queried)  -> {c['path']}")
    else:
        print(f"  would create {len(report['wouldCreate'])} draft stub(s) (dry-run; pass --write):")
        for c in report["wouldCreate"]:
            print(f"    ? {c['id']}  ({c['hits']}× queried)")
    if report["rejected"]:
        print(f"  rejected {len(report['rejected'])} (gate failure):")
        for r in report["rejected"]:
            print(f"    x {r['id']}: {r['reasons']}")
    if report["enrichTargets"]:
        print(f"  {len(report['enrichTargets'])} existing page(s) flagged for enrichment:")
        for e in report["enrichTargets"][:10]:
            print(f"    ~ {e['target']}  ({e['hits']}× came up short)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
