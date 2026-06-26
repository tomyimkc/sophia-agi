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
from agent.factcheck_oracle import GoogleFactCheckOracle, compose_live_factcheck  # noqa: E402
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
                    help="force the keyed Google Fact Check Tools oracle (error if GOOGLE_FACTCHECK_API_KEY is unset)")
    ap.add_argument("--no-google-factcheck", action="store_true",
                    help="opt out of the oracle even when a key is present")
    ap.add_argument("--google-page-size", type=int, default=10)
    ap.add_argument("--nli", action="store_true",
                    help="use an LLM NLI entailment for ClaimReview sources instead of the lexical screen (needs DEEPSEEK_API_KEY)")
    ap.add_argument("--target-fabrication-rate", type=float, default=0.01)
    args = ap.parse_args(argv)

    rows = load_jsonl(args.pack)
    if args.live:
        backend = LiveFactBackend()
    else:
        backend = FixtureFactBackend.from_file(args.fixtures)

    # The Google oracle is part of the DEFAULT live composition: on automatically
    # when a key is present (offline/CI with no key is unchanged). --google-factcheck
    # forces it (error if no key); --no-google-factcheck opts out.
    oracle = None
    if not args.no_google_factcheck:
        candidate = GoogleFactCheckOracle(page_size=args.google_page_size)
        if args.google_factcheck and not candidate.enabled:
            print("ERROR: --google-factcheck set but GOOGLE_FACTCHECK_API_KEY is not in the environment", file=sys.stderr)
            return 2
        if candidate.enabled:
            oracle = candidate

    factcheck_entailment = None
    if args.nli:
        from agent.factcheck_nli import NLIEntailment
        factcheck_entailment = NLIEntailment(source_types={"factcheck"})

    retriever, entailment, oracle_active = compose_live_factcheck(
        backend.retriever, backend.entailment,
        oracle=oracle, factcheck_entailment=factcheck_entailment,
    )
    live_backend = bool(args.live) or oracle_active
    if oracle_active:
        print(f"[factcheck] Google ClaimReview oracle ACTIVE (nli={'on' if args.nli else 'off'})", file=sys.stderr)

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
