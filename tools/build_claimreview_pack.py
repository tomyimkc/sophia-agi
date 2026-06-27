#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build a third-party-grounded benchmark pack from the Google Fact Check Tools API.

The repo's binding constraint is third-party independence: every existing
benchmark (provenance, calibration) is self-authored. This tool harvests REAL
professional fact-check verdicts (ClaimReview markup from AP, Reuters, Snopes,
PolitiFact, Full Fact, AFP, BBC, ...) via the Google Fact Check Tools API and
normalizes them into Sophia's Case schema, producing the repo's FIRST
non-self-authored pack.

Domain note (recorded honestly): the Fact Check API covers CONTEMPORARY claims
(vaccines, climate, politics, science misconceptions) — it returns ~0 claims for
historical authorship misattribution (Confucius, Homer, etc.). So this pack is a
NEW capability axis (contemporary-claim verification), NOT validation of the
existing dolphin provenance delta. The two are honestly separate.

Output: provenance_bench/data/claimreview_pack.json — {id, claim, label,
gold_verdict, publisher, rating_raw, source_url, language, query}.

Labels (normalized from textualRating): "false" / "true" / "mixed" / "other".
Only "false" and "true" feed the eval (clean ground truth); "mixed"/"other" are
retained for audit but excluded from scoring.

No overclaim: this is a harvested pack, scored against external professional
verdicts. canClaimAGI stays False.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "provenance_bench" / "data" / "claimreview_pack.json"

BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Query sets. The API's coverage is heavily biased toward already-viral claims,
# which models ALSO know from training data — so the "famous" pack (default) is
# the WRONG substrate for showing grounding value (models reject those raw). The
# "obscure"/recent set targets viral-but-RECENT (2024-2026) and specific
# misinformation that a model with an older training cutoff might genuinely
# endorse — the right substrate for a non-null Δ. Select via --set.
QUERY_SETS: dict[str, list[str]] = {
    "famous": [
        # health / vaccines
        "vaccines cause autism", "covid vaccine microchip", "5G coronavirus",
        "hydroxychloroquine covid cure", "fluoride water IQ", "mmr vaccine",
        "covid vaccine infertility", "ivermectin covid", "vaccine shedding",
        "covid death count exaggerated",
        # climate / energy
        "climate change hoax", "co2 not greenhouse gas", "climate models inaccurate",
        "solar panels cause cancer", "wind turbines bird extinction",
        # politics / elections (generic)
        "election fraud", "mail ballot fraud", "immigrant crime statistics",
        "inflation cause", "crime rate rising",
        # science misconceptions
        "Einstein failed math", "Great Wall visible from space", "humans use 10 percent of brain",
        "sugar makes children hyperactive", "cracking knuckles arthritis",
        "shaving makes hair grow back thicker", "water spins drain hemisphere",
        "bull sees red color", "goldfish three second memory",
        # history misconceptions
        "Marie Antoinette let them eat cake", "Vikings horned helmets",
        "Columbus proved earth round", "Washington wooden teeth",
        "pyramids built by aliens", "mao sparrows campaign",
        # tech / AI / media
        "deepfake election", "5g tracking", "ai replaces jobs", "phone cause cancer",
        "wifi radiation harmful", "vpn makes anonymous",
    ],
    "obscure": [
        # 2024-2026 viral misinfo (recent; older-cutoff models may not know)
        "hurricane helene fema response 2024", "katrina fema money immigrants",
        "springfield haitian immigrants pets 2024", "trump Springfield comment",
        "israel gaza hospital 2024", "unrwa staff october 7",
        "taylor swift endorse 2024", "ai generated trump pope photo 2025",
        "trump crypto wallet 2025", "melania trump body double",
        "maui fire directed energy weapon", "fema 750 loan hurricane",
        "california fire aid 180 days", "los angeles hollywood aid",
        # specific numerics (training-thin)
        "us inflation rate 2024", "gdp growth 2024",
        "border crossings 2024 statistics", "fentanyl deaths 2024",
        "ev sales decline 2024", "tesla robotaxi 2024",
        # recent science/health specifics
        "bird flu h5n1 transmission human 2024", "mpox 2024 outbreak",
        "semaglutide pancreatic cancer", "pfizer ceo resign 2024",
        "seed oils inflammation study",
        # niche conspiracies (less canonical than the famous-pack ones)
        "chemtrails weather modification", "haarp earthquake",
        "flat earth antarctica ice wall", "moon landing fake",
        # tech/specific
        "openai nonprofit 2024", "sam altman fired timeline",
    ],
}
QUERIES = QUERY_SETS["famous"]  # default, backward-compatible

# Rating normalization. The API's textualRating is free-form across languages;
# map it to a clean {true, false, mixed, other} label. Conservative: anything
# ambiguous -> "other" (excluded from scoring), never forced to true/false.
# Order matters: negated forms ("not true", "not false", "this is untrue") are
# checked FIRST so the bare-word regexes below don't fire on the embedded token.
_NEG_TRUE = re.compile(r"\b(not|n'?t|isn'?t|no|un|non)[- ]?(true|correct|accurate)\b", re.I)
_NEG_FALSE = re.compile(r"\b(not|n'?t|isn'?t|no|un|non)[- ]?(false|wrong|incorrect)\b", re.I)
_FALSE = re.compile(r"\b(false|falso|falsch|faux|fake|wrong|incorrect|untrue|misleading|distort|debunk|myth|disputed by experts|cherry pick|altered|fabricat|unsupported|no evidence|baseless|flawed|lacks context|pinocchio)\b", re.I)
_TRUE = re.compile(r"^(true|correct|accurate|verified|mostly true|largely true|half true)\b|\bthis is true\b", re.I)
_MIXED = re.compile(r"\b(mixed|partly|partially|half|części|misleading but|mostly false but|some truth|half true|mostly true)\b", re.I)


def normalize_rating(raw: str) -> str:
    r = (raw or "").strip().lower()
    if not r:
        return "other"
    # Negation must be resolved before bare-word matching.
    if _NEG_TRUE.search(r):
        return "false"
    if _NEG_FALSE.search(r):
        return "true"
    if _MIXED.search(r):
        return "mixed"
    if _FALSE.search(r):
        return "false"
    if _TRUE.search(r):
        return "true"
    return "other"


def search(q: str, key: str, page_token: str | None = None) -> dict:
    p = {"query": q, "key": key, "pageSize": 20, "languageCode": "en"}
    if page_token:
        p["pageToken"] = page_token
    url = BASE + "?" + urllib.parse.urlencode(p)
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def harvest(key: str, queries: list[str] | None = None, per_query_pages: int = 1) -> list[dict]:
    """Harvest claims; dedup by (claim text, publisher). Returns raw claim rows."""
    if queries is None:
        queries = QUERIES
    seen: set[str] = set()
    rows: list[dict] = []
    for q in queries:
        token = None
        for _ in range(per_query_pages):
            try:
                d = search(q, key, token)
            except Exception as e:
                print(f"  [{q}] error: {e}", file=sys.stderr)
                break
            for c in d.get("claims", []):
                cr = (c.get("claimReview") or [{}])[0]
                text = (c.get("text") or "").strip()
                pub = (cr.get("publisher") or {}).get("name", "?")
                dedup_key = hashlib.sha1(f"{text}|{pub}".encode()).hexdigest()[:16]
                if not text or dedup_key in seen:
                    continue
                seen.add(dedup_key)
                rows.append({
                    "claim": text,
                    "claimant": c.get("claimant"),
                    "claimDate": c.get("claimDate"),
                    "publisher": pub,
                    "publisherSite": (cr.get("publisher") or {}).get("site"),
                    "rating_raw": cr.get("textualRating"),
                    "rating_normalized": normalize_rating(cr.get("textualRating") or ""),
                    "source_url": cr.get("url"),
                    "language": cr.get("languageCode"),
                    "reviewDate": (cr.get("reviewDate") or (cr.get("claimReview") or [{}])[0].get("reviewDate") if False else None),
                    "query": q,
                })
            token = d.get("nextPageToken")
            if not token:
                break
            time.sleep(0.4)
        print(f"  [{len([r for r in rows if r['query']==q]):>2} cum] {q}")
    return rows


def build_pack(rows: list[dict]) -> dict:
    """Assemble the pack JSON with provenance + honest meta."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r["rating_normalized"], []).append(r)
    return {
        "_meta": {
            "schema": "sophia.claimreview_pack.v1",
            "description": (
                "Third-party-grounded benchmark pack harvested from the Google Fact Check "
                "Tools API (ClaimReview markup from AP/Reuters/Snopes/PolitiFact/AFP/BBC/...). "
                "The repo's FIRST non-self-authored pack. Domain: contemporary claims "
                "(vaccines/climate/politics/science+history misconceptions) — NOT historical "
                "authorship, so this is a NEW capability axis, not validation of the dolphin "
                "provenance delta. Labels normalized from free-form textualRating; only "
                "'false'/'true' feed eval scoring (clean ground truth); 'mixed'/'other' "
                "retained for audit, excluded from scoring."
            ),
            "source": "Google Fact Check Tools API (v1alpha1/claims:search)",
            "groundTruth": "external professional fact-checker verdicts (ClaimReview)",
            "candidateOnly": True,
            "canClaimAGI": False,
            "labelCounts": {k: len(v) for k, v in sorted(by_label.items())},
            "nPublishers": len({r["publisher"] for r in rows}),
            "harvestedAt": time.strftime("%Y-%m-%d", time.gmtime()),
        },
        "claims": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key-file", default="/tmp/.gfc", help="file containing the Google Fact Check API key")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--pages", type=int, default=1, help="result pages per query (each ~20 claims)")
    ap.add_argument("--set", default="famous", choices=list(QUERY_SETS),
                    help="'famous' (canonical, models know them) or 'obscure' (recent/niche, "
                         "models may genuinely endorse — the right substrate for a non-null Δ)")
    args = ap.parse_args(argv)

    key = Path(args.key_file).read_text().strip()
    queries = QUERY_SETS[args.set]
    print(f"harvesting set={args.set!r}: {len(queries)} queries x {args.pages} page(s)...")
    rows = harvest(key, queries=queries, per_query_pages=args.pages)
    pack = build_pack(rows)
    OUT_PATH = Path(args.out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pack, ensure_ascii=False, indent=2), encoding="utf-8")

    lc = pack["_meta"]["labelCounts"]
    print(f"\npack written -> {OUT_PATH}")
    print(f"  total claims: {len(rows)}  |  publishers: {pack['_meta']['nPublishers']}")
    print(f"  labels: {lc}")
    print(f"  eval-usable (false+true): {lc.get('false',0)+lc.get('true',0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
