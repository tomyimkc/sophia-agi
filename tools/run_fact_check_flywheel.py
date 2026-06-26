#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Turn accepted fact-check learning candidates through quarantine + recheck."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_flywheel import append_jsonl, extract_learning_candidates, load_json, run_flywheel_from_report  # noqa: E402
from agent.live_sources import FixtureFactBackend, GoogleFactCheckBackend, LiveFactBackend  # noqa: E402
from tools.run_fact_check_live_eval import _compose  # noqa: E402

DEFAULT_REPORT = ROOT / "agi-proof" / "fact-check-live" / "fact-check-live-eval.public-report.json"
DEFAULT_FIXTURES = ROOT / "eval" / "fact_check" / "fixtures_v1.json"
DEFAULT_QUARANTINE = ROOT / "agi-proof" / "fact-check-live" / "learning-candidates.quarantine.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "fact-check-live" / "flywheel.public-report.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES))
    ap.add_argument("--quarantine", default=str(DEFAULT_QUARANTINE))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--live", action="store_true")
    ap.add_argument(
        "--google-factcheck", action="store_true",
        help="add Google Fact Check Tools (ClaimReview) evidence; reads GOOGLE_FACTCHECK_API_KEY. Composes with --live.",
    )
    args = ap.parse_args(argv)
    report = load_json(args.report)
    backends = [LiveFactBackend() if args.live else FixtureFactBackend.from_file(args.fixtures)]
    if args.google_factcheck:
        g = GoogleFactCheckBackend()
        if g.api_key:
            backends.append(g)
        else:
            print("WARNING: --google-factcheck set but GOOGLE_FACTCHECK_API_KEY is empty; skipped.", file=sys.stderr)
    backend = _compose(*backends)
    candidates = extract_learning_candidates(report)
    append_jsonl(args.quarantine, candidates)
    flywheel = run_flywheel_from_report(report, retriever=backend.retriever, entailment=backend.entailment,
                                        doi_resolver=backend.doi_resolver, url_resolver=backend.url_resolver)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(flywheel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "quarantine": args.quarantine, "nCandidates": flywheel["nCandidates"],
                      "nPromotedProvisional": flywheel["nPromotedProvisional"],
                      "canonicalWikiWrite": flywheel["canonicalWikiWrite"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
