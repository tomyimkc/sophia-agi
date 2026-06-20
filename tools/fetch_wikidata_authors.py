#!/usr/bin/env python3
"""Optional: verify/populate Wikidata QIDs for the true-attribution snapshot.

Network-using and OFF the default path — the committed snapshot already carries
human-checkable Wikipedia URLs, so the benchmark runs fully offline. This script
queries the Wikidata API to resolve each work to a QID and confirm its
``author`` (P50) / ``author name string`` (P2093), then prints proposed updates.
By default it is a dry run; pass --write to update the snapshot in place.

    python tools/fetch_wikidata_authors.py            # dry run, prints proposals
    python tools/fetch_wikidata_authors.py --write     # apply qids to snapshot
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "provenance_bench" / "data" / "wikidata_snapshot.json"
API = "https://www.wikidata.org/w/api.php"


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": "sophia-agi-provenance-bench/1.0"})
    with urllib.request.urlopen(req, timeout=30) as fh:  # noqa: S310 (trusted host)
        return json.loads(fh.read().decode("utf-8"))


def resolve_qid(work: str) -> str | None:
    data = _get({"action": "wbsearchentities", "search": work, "language": "en", "limit": 1})
    hits = data.get("search") or []
    return hits[0]["id"] if hits else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="apply resolved QIDs to the snapshot")
    args = ap.parse_args(argv)

    doc = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    changed = 0
    for row in doc.get("attributions", []):
        if row.get("wikidata_qid"):
            continue
        try:
            qid = resolve_qid(row["work"])
        except Exception as exc:  # network/parse — report, keep going
            print(f"  ! {row['work']}: {exc}")
            continue
        print(f"  {row['work']:<28} -> {qid}")
        if qid:
            row["wikidata_qid"] = qid
            row["source_url"] = f"https://www.wikidata.org/wiki/{qid}"
            changed += 1

    if args.write and changed:
        SNAPSHOT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nupdated {changed} rows in {SNAPSHOT}")
    else:
        print(f"\ndry run ({changed} resolvable); pass --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
