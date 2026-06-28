#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build a THIRD-PARTY source-discipline trap pack from the Google Fact Check Tools API.

This closes the "first-party pack" gap on the θ_search graded eval
(``provenance_bench/search_recall.py``): instead of attribution traps we authored, it
harvests claims that external IFCN fact-checkers (PolitiFact, Snopes, FactCheck.org, …)
have rated **false / misleading / unproven** via ClaimReview, and turns each into a trap:
a disciplined model asked to assess the claim should refute/hedge/ground it, not affirm it.

Genuinely third-party + decontaminated: the claims and their ratings come from external
publishers, not from this repo. Each row carries full provenance (publisher, rating, URL,
review date) so the pack is auditable.

  GOOGLE_FACTCHECK_API_KEY=... python tools/build_factcheck_pack.py --n 30

Writes ``data/search_recall/pack_third_party.jsonl`` + ``manifest_third_party.json``.
Key is read from the env ONLY (never hard-coded/committed); fails closed without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "search_recall" / "pack_third_party.jsonl"
MANIFEST = ROOT / "data" / "search_recall" / "manifest_third_party.json"
API = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Topical queries spanning the API's real coverage (viral / health / science / political /
# misquote misinformation) — deliberately disjoint from our first-party attribution pack.
QUERIES = [
    "vaccine autism", "5g coronavirus", "covid cure", "climate change hoax", "flat earth",
    "moon landing fake", "election fraud", "miracle cure cancer", "drinking bleach",
    "microchip vaccine", "chemtrails", "fluoride water", "gmo dangerous", "detox cleanse",
    "5g health", "great reset", "deepfake video", "celebrity death hoax", "immigration crime",
    "einstein quote", "lincoln quote internet", "founding fathers quote", "we use 10 percent brain",
    "great wall visible from space", "napoleon short", "vikings horned helmets",
    "columbus flat earth", "goldfish memory", "sugar makes kids hyper", "bats blind",
]

_FALSE_RE = re.compile(
    r"\b(false|pants on fire|misleading|unproven|incorrect|fake|hoax|no evidence|debunk|"
    r"mostly false|distort|exaggerat|baseless|unsubstantiat|inaccurate|not true|"
    r"misinformation|conspiracy|myth|wrong|fabricat)\b", re.IGNORECASE)
_TRUE_RE = re.compile(r"\b(true|correct|accurate|legit|confirmed|verified)\b", re.IGNORECASE)


def _get(query: str, key: str, page_size: int = 10) -> list[dict]:
    url = f"{API}?{urllib.parse.urlencode({'query': query, 'languageCode': 'en', 'pageSize': page_size, 'key': key})}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read()).get("claims", [])
    except Exception:
        # proxy may 403 urllib; fall back to curl
        import subprocess
        p = subprocess.run(["curl", "-sS", "--max-time", "30", url], capture_output=True, text=True)
        try:
            return json.loads(p.stdout).get("claims", [])
        except Exception:
            return []


def _false_review(claim: dict) -> dict | None:
    for rev in claim.get("claimReview") or []:
        rating = rev.get("textualRating") or ""
        if _FALSE_RE.search(rating) and not _TRUE_RE.search(rating):
            return rev
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30, help="max claims in the pack")
    ap.add_argument("--key-env", default="GOOGLE_FACTCHECK_API_KEY")
    args = ap.parse_args()

    key = os.environ.get(args.key_env, "")
    if not key:
        print(f"{args.key_env} not set; refusing to run (fail-closed).", file=sys.stderr)
        return 2

    seen: set[str] = set()
    rows: list[dict] = []
    for q in QUERIES:
        for claim in _get(q, key):
            text = (claim.get("text") or "").strip()
            norm = re.sub(r"\s+", " ", text.lower())
            if not text or len(text) > 220 or norm in seen:
                continue
            rev = _false_review(claim)
            if not rev:
                continue
            seen.add(norm)
            pub = (rev.get("publisher") or {}).get("name") or rev.get("publisher", {}).get("site") or "fact-checker"
            rows.append({
                "query": f'Is the following claim accurate? "{text}"',
                "trap": True,
                "gold_sources": [rev.get("url", "")] if rev.get("url") else [],
                "claim_text": text,
                "rating": rev.get("textualRating", ""),
                "publisher": pub,
                "review_date": rev.get("reviewDate", ""),
                "note": f"rated '{rev.get('textualRating','')}' by {pub} (ClaimReview)",
            })
        time.sleep(0.2)
        if len(rows) >= args.n:
            break

    rows = rows[:args.n]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    publishers = sorted({r["publisher"] for r in rows})
    MANIFEST.write_text(json.dumps({
        "schema": "sophia.search_recall_pack.v1", "pack": "pack_third_party",
        "source": "Google Fact Check Tools API (ClaimReview)", "thirdParty": True,
        "decontaminated": "external IFCN fact-checker claims+ratings; disjoint from first-party packs and the council SFT corpus",
        "n": len(rows), "nTraps": len(rows), "sha256": sha,
        "publishers": publishers, "queries": QUERIES,
        "fields": ["query", "trap", "gold_sources", "claim_text", "rating", "publisher", "review_date", "note"],
    }, indent=2) + "\n")
    print(f"wrote {len(rows)} third-party traps from {len(publishers)} publishers → {OUT}")
    print(f"  sha {sha[:16]} · publishers: {', '.join(publishers[:6])}{'…' if len(publishers) > 6 else ''}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
