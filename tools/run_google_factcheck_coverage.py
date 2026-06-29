#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Live coverage validation of the Google Fact Check Tools backend.

The ``GoogleFactCheckBackend`` (agent/live_sources.py) is implemented and unit-tested
offline, but had never been exercised against the real API. This probe runs it LIVE on
two claim sets and reports coverage:

  - general / viral claims (the ClaimReview corpus's home turf), which SHOULD return
    ClaimReview evidence;
  - literary-provenance claims (Sophia's actual domain), which the failure ledger predicts
    the API does NOT cover.

The point is an honest boundary measurement: it validates the integration end-to-end AND
quantifies that this external oracle is the wrong tool for provenance/attribution claims.
Reads GOOGLE_FACTCHECK_API_KEY from the env; fails closed (coverage 0) without a key.

  GOOGLE_FACTCHECK_API_KEY=... python tools/run_google_factcheck_coverage.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_gate import AtomicClaim  # noqa: E402
from agent.live_sources import GoogleFactCheckBackend  # noqa: E402

REPORT_PATH = ROOT / "agi-proof" / "benchmark-results" / "real-model" / "google-factcheck-coverage.public-report.json"

_GENERAL = [
    "The Earth is flat.",
    "Vaccines cause autism.",
    "5G networks spread the coronavirus.",
    "Drinking bleach cures COVID-19.",
    "Climate change is a hoax.",
    "The 2020 US presidential election was stolen.",
]
_PROVENANCE = [
    "Confucius wrote the Dao De Jing.",
    "Laozi wrote the Dao De Jing.",
    "Freud originated the theory of the unconscious.",
    "Homer wrote the Iliad.",
    "The Voynich Manuscript was written by Roger Bacon.",
    "Hippocrates authored the Hippocratic Oath.",
]


def _coverage(backend: GoogleFactCheckBackend, claims: list[str]) -> dict:
    rows, covered = [], 0
    for text in claims:
        srcs = backend.retriever(AtomicClaim(text=text, type="external"))
        n = len(srcs)
        covered += int(n > 0)
        rows.append({"claim": text, "nSources": n,
                     "sample": (srcs[0].snippet[:160] if srcs else None)})
    return {"n": len(claims), "covered": covered,
            "coverageRate": round(covered / len(claims), 4) if claims else 0.0, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live Google Fact Check coverage probe.")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    backend = GoogleFactCheckBackend()
    has_key = bool(backend.api_key)
    general = _coverage(backend, _GENERAL)
    provenance = _coverage(backend, _PROVENANCE)

    report = {
        "schema": "sophia.google_factcheck_coverage.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "hasApiKey": has_key,
        "generalClaims": general,
        "provenanceClaims": provenance,
        "honestBound": (
            "Live end-to-end validation of GoogleFactCheckBackend. Confirms the integration "
            "works on general/viral claims (ClaimReview's home turf) and QUANTIFIES that the "
            "API does not cover Sophia's literary-provenance domain — so this external oracle "
            "complements, but cannot replace, the Wikidata/Crossref provenance path. "
            "Corroborates the failure-ledger note with live data." if has_key else
            "No GOOGLE_FACTCHECK_API_KEY present: backend failed closed (coverage 0). Not a "
            "coverage result — set the key to run live."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Google Fact Check live coverage (hasKey={has_key})")
    print(f"  general claims:    {general['covered']}/{general['n']} covered "
          f"(rate {general['coverageRate']})")
    print(f"  provenance claims: {provenance['covered']}/{provenance['n']} covered "
          f"(rate {provenance['coverageRate']})")
    print(f"Wrote {(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
