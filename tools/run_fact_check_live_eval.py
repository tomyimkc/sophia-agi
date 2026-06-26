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
from agent.live_sources import (  # noqa: E402
    FixtureFactBackend,
    GoogleFactCheckBackend,
    LiveFactBackend,
)

DEFAULT_PACK = ROOT / "eval" / "fact_check" / "heldout_v1.jsonl"
DEFAULT_FIXTURES = ROOT / "eval" / "fact_check" / "fixtures_v1.json"
DEFAULT_OUT = ROOT / "agi-proof" / "fact-check-live" / "fact-check-live-eval.public-report.json"


def _compose(*backends) -> "object":
    """Compose N backends into one: retrieve from all, entail per-source (the
    originating backend knows its own source's relation), resolve via the first
    backend that has a resolver. Lets ``--google-factcheck`` ADD an independent
    evidence family alongside ``--live`` rather than replace it."""
    backends = [b for b in backends if b is not None]
    if len(backends) == 1:
        return backends[0]

    class _Composite:
        def retriever(self, claim):
            from agent.live_sources import ranked_sources
            out = []
            for b in backends:
                out.extend(b.retriever(claim))
            return ranked_sources(out)

        def entailment(self, claim, source):
            # Each source encodes its relation in its id (fixture/google) or via
            # the structured-entailment helpers (live). Try the backend whose
            # source_type matches the source first, then fall back to any.
            st = (getattr(source, "source_type", "") or "").lower()
            order = backends
            if st:
                order = sorted(backends, key=lambda b: 0 if (st == "google_factcheck" and isinstance(b, GoogleFactCheckBackend)) else 1)
            for b in order:
                rel = b.entailment(claim, source)
                if rel != "irrelevant":
                    return rel
            return "irrelevant"

        def doi_resolver(self, doi):
            return backends[0].doi_resolver(doi)

        def url_resolver(self, url):
            return backends[0].url_resolver(url)

    return _Composite()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(DEFAULT_PACK), help="held-out JSONL pack")
    ap.add_argument("--fixtures", default=str(DEFAULT_FIXTURES), help="offline fixture JSON")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="report output path")
    ap.add_argument("--live", action="store_true", help="use keyless live Wikidata/Crossref/URL/macro/scholarly backend")
    ap.add_argument(
        "--google-factcheck", action="store_true",
        help="add Google Fact Check Tools (ClaimReview) as an evidence family; "
             "reads GOOGLE_FACTCHECK_API_KEY from the env. Composes with --live.",
    )
    ap.add_argument("--condition", choices=["full", "raw"], default="full",
                    help="full = gate+retrieval pipeline (default); raw = base model alone (no gate), "
                         "the baseline the gate is measured against")
    ap.add_argument("--model", default=None, help="model spec for --condition raw (e.g. mlx:Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--target-fabrication-rate", type=float, default=0.01)
    args = ap.parse_args(argv)

    rows = load_jsonl(args.pack)

    if args.condition == "raw":
        # RAW arm: base model classifies each claim with NO gate/retrieval. Same pack, same
        # external labels -> raw-vs-full isolates the gate's fabrication-reduction value.
        if not args.model:
            print("--condition raw requires --model <spec>", file=sys.stderr)
            return 2
        from agent.model import default_client  # noqa: E402
        from agent.raw_fact_classifier import raw_fact_verdict  # noqa: E402

        client = default_client(args.model)
        report = run_fact_check_eval(
            rows,
            live_backend=bool(args.live),
            target_fabrication_rate=args.target_fabrication_rate,
            verdict_fn=lambda row: raw_fact_verdict(row, client),
        )
        report["condition"] = "raw"
        report["rawModel"] = args.model
    else:
        backends = []
        backends.append(LiveFactBackend() if args.live else FixtureFactBackend.from_file(args.fixtures))
        google_used = False
        if args.google_factcheck:
            g = GoogleFactCheckBackend()
            if g.api_key:
                backends.append(g)
                google_used = True
            else:
                print("WARNING: --google-factcheck set but GOOGLE_FACTCHECK_API_KEY is empty; "
                      "Google backend skipped (fail-closed).", file=sys.stderr)
        backend = _compose(*backends)
        report = run_fact_check_eval(
            rows,
            retriever=backend.retriever,
            entailment=backend.entailment,
            doi_resolver=backend.doi_resolver,
            url_resolver=backend.url_resolver,
            live_backend=bool(args.live),
            target_fabrication_rate=args.target_fabrication_rate,
        )
        report["condition"] = "full"
        report["googleFactCheckBackend"] = google_used
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
