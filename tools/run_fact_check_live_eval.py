#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the measured out-of-wiki fact-check evaluation.

Default mode is deterministic/offline: it uses committed held-out claims plus
committed source fixtures. ``--live`` switches DOI/URL/Wikidata/macro/scholarly adapters to
keyless network backends; live results are still NOT Level-3 AGI evidence unless
run under the hidden-reviewer protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_eval import load_jsonl, run_fact_check_eval, write_report  # noqa: E402
from agent.factcheck_oracle import (  # noqa: E402
    GoogleFactCheckOracle,
    combined_retriever,
    dispatched_entailment,
)
from agent.live_sources import FixtureFactBackend, LiveFactBackend  # noqa: E402

DEFAULT_PACK = ROOT / "eval" / "fact_check" / "heldout_v1.jsonl"
DEFAULT_FIXTURES = ROOT / "eval" / "fact_check" / "fixtures_v1.json"
DEFAULT_OUT = ROOT / "agi-proof" / "fact-check-live" / "fact-check-live-eval.public-report.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(DEFAULT_PACK), help="held-out JSONL pack")
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES), help="offline fixture JSON")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="report output path")
    ap.add_argument("--live", action="store_true", help="use keyless live Wikidata/Crossref/URL/macro/scholarly backend")
    ap.add_argument("--google-factcheck", action="store_true",
                    help="compose the keyed Google Fact Check Tools oracle (needs GOOGLE_FACTCHECK_API_KEY)")
    ap.add_argument("--google-page-size", type=int, default=10)
    ap.add_argument("--target-fabrication-rate", type=float, default=0.01)
    args = ap.parse_args(argv)

    rows = load_jsonl(args.pack)
    if args.live:
        backend = LiveFactBackend()
    else:
        backend = FixtureFactBackend.from_file(args.fixtures)

    retriever = backend.retriever
    entailment = backend.entailment
    live_backend = bool(args.live)
    if args.google_factcheck:
        oracle = GoogleFactCheckOracle(page_size=args.google_page_size)
        if not oracle.enabled:
            print("ERROR: --google-factcheck set but GOOGLE_FACTCHECK_API_KEY is not in the environment", file=sys.stderr)
            return 2
        retriever = combined_retriever([backend.retriever, oracle.retriever])
        entailment = dispatched_entailment(oracle, backend.entailment)
        live_backend = True  # a keyed network oracle is a live backend

    report = run_fact_check_eval(
        rows,
        retriever=retriever,
        entailment=entailment,
        doi_resolver=backend.doi_resolver,
        url_resolver=backend.url_resolver,
        live_backend=live_backend,
        target_fabrication_rate=args.target_fabrication_rate,
    )
    write_report(report, args.out)
    print(json.dumps({
        "out": args.out,
        "candidateOnly": report["candidateOnly"],
        "liveBackendUsed": report["liveBackendUsed"],
        "n": report["n"],
        "metrics": report["metrics"],
        "derivedFloors": report["derivedFloors"],
    }, indent=2, ensure_ascii=False))
    return 0 if report["metrics"]["fabricationRate"] <= args.target_fabrication_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
