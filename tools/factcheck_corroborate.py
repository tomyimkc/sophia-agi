#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent external corroboration for the corpus's DEBUNKABLE (pop-myth) claims via the
Google Fact Check Tools API (ClaimReview markup from Snopes/PolitiFact/etc.).

This adds a third-party verification layer for the slice of the corpus that published fact-checkers
actually cover — popular myths. It is NOT a verifier for the source-discipline CORE (authorship /
provenance / tradition-separation): fact-checkers do not review "who wrote the Dao De Jing." So this
corroborates one honest slice and the coverage gap is itself reported, not hidden.

    GOOGLE_FACTCHECK_API_KEY=... python3 tools/factcheck_corroborate.py
Writes agi-proof/benchmark-results/wisdom-market/factcheck-corroboration.json. No key -> dry list.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "factcheck-corroboration.json"
API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# The corpus's debunkable myth claims (mirrors data/history_events.json myths + the pop_myth
# benchmark family), phrased as the FALSE popular belief the corpus marks as a myth.
MYTHS = [
    "The Great Wall of China is visible from space with the naked eye",
    "Marco Polo introduced pasta to Italy",
    "Napoleon was unusually short",
    "Medieval Europeans believed the earth was flat",
    "Vikings wore horned helmets in battle",
    "Humans only use 10 percent of their brain",
    "People have distinct left-brained or right-brained personalities",
    "Students learn better when taught in their preferred learning style",
    "The tongue has separate zones for different tastes",
    "Hair and fingernails keep growing after death",
    "Einstein failed mathematics in school",
    "Christopher Columbus proved the earth was round",
    "Thomas Edison invented the light bulb",
    "Bulls are enraged by the color red",
    "Bats are blind",
]
# A few source-discipline CORE claims, to demonstrate (honestly) the coverage gap.
CORE = [
    "Laozi wrote the Dao De Jing",
    "Homer wrote the Iliad",
    "The Yijing was written by a single author",
]


def _search(query: str, key: str) -> list:
    url = API + "?" + urllib.parse.urlencode({"query": query, "languageCode": "en", "key": key})
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as exc:  # never crash the whole sweep on one query
        return [{"error": repr(exc)}]
    out = []
    for c in data.get("claims", [])[:3]:
        for rev in c.get("claimReview", [])[:1]:
            out.append({"claimText": c.get("text", "")[:160], "publisher": (rev.get("publisher") or {}).get("name"),
                        "rating": rev.get("textualRating"), "url": rev.get("url"), "reviewDate": rev.get("reviewDate")})
    return out


def main() -> int:
    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "").strip()
    if not key:
        print("No GOOGLE_FACTCHECK_API_KEY set. Myth claims that WOULD be checked:")
        for m in MYTHS:
            print("  -", m)
        return 0
    myth_results, core_results = [], []
    covered = 0
    for m in MYTHS:
        revs = _search(m, key)
        has = bool(revs and "error" not in revs[0] and revs[0].get("rating"))
        if has:
            covered += 1
        myth_results.append({"claim": m, "covered": has, "factChecks": revs})
        print(f"  [{'✓' if has else '·'}] {m[:60]:60s} {revs[0].get('rating','—') if has else '(no published fact-check)'}")
    for c in CORE:
        revs = _search(c, key)
        has = bool(revs and "error" not in revs[0] and revs[0].get("rating"))
        core_results.append({"claim": c, "covered": has, "factChecks": revs})

    report = {
        "source": "Google Fact Check Tools API (claims:search; ClaimReview markup)",
        "scope": ("Independent third-party corroboration for the corpus's DEBUNKABLE pop-myth slice "
                  "ONLY. Published fact-checkers do not review authorship/provenance, so the "
                  "source-discipline CORE is necessarily uncovered — reported, not hidden."),
        "mythCoverage": {"checked": len(MYTHS), "corroborated": covered,
                         "rate": round(covered / len(MYTHS), 3)},
        "coreCoverage": {"checked": len(CORE),
                         "corroborated": sum(1 for r in core_results if r["covered"]),
                         "note": "expected ~0 — fact-checkers do not cover literary attribution"},
        "myths": myth_results, "core": core_results,
        "boundary": "external corroboration of one slice; not a claim about the core; candidate_only",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nMYTH corroboration: {covered}/{len(MYTHS)} ({report['mythCoverage']['rate']}) have a published "
          f"third-party fact-check.\nCORE corroboration: {report['coreCoverage']['corroborated']}/{len(CORE)} "
          f"(expected ~0 — coverage gap is honest).\nwrote -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
