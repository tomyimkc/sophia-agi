#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate agi-proof/failure-ledger.md structurally and summarize OPEN vs RESOLVED.

The failure ledger is the most honest artifact in the repo — it records where the system
is NOT AGI. This tool keeps it from silently rotting: every table entry must have an id,
a Status, and a Claim impact; and it emits an OPEN/CLOSED summary so the evidence manifest
can surface "what is still blocking the AGI claim".

The ledger has two shapes:
  - a Markdown table: ``| <id> | <Status> | <Claim impact> | <Required response> |``
  - appended ``## <id>`` detailed sections whose first ``**Status:** <STATUS>`` line is the
    authoritative status (these override/extend the table for items that got a write-up).

    python tools/validate_failure_ledger.py            # validate + print OPEN/CLOSED summary
    python tools/validate_failure_ledger.py --check    # exit 1 if structurally invalid
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "agi-proof" / "failure-ledger.md"

_OPEN_MARKERS = ("open", "partial", "blocked", "not yet", "pending")
_RESOLVED_MARKERS = ("closed", "cleared", "superseded", "resolved", "fixed", "falsified", "complete")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def classify(status: str) -> str:
    """Map a free-form status string to open | resolved | other."""
    s = _norm(status).lower()
    if not s:
        return "other"
    # 'closed'/'resolved' must beat 'open' (a row can't be both, but be safe)
    if any(m in s for m in _RESOLVED_MARKERS):
        return "resolved"
    if any(m in s for m in _OPEN_MARKERS):
        return "open"
    return "other"


def _parse_table(lines: list[str]) -> list[dict]:
    """Parse the main failure-ledger table (the one at the top, before the first ``##``
    detailed section). Sub-tables inside ``##`` write-ups (run tables, channel tables) are
    deliberately NOT parsed — they are not failure-ledger rows. Skips header/separator."""
    rows: list[dict] = []
    # the ledger table lives above the first '## ' detailed section
    cutoff = next((i for i, ln in enumerate(lines) if ln.startswith("## ")), len(lines))
    for line in lines[:cutoff]:
        st = line.strip()
        if not st.startswith("|") or not st.endswith("|"):
            continue
        cells = [_norm(c) for c in st.strip("|").split("|")]
        if len(cells) < 4:
            continue
        # skip separator row (---) and the header row
        if re.fullmatch(r"[\s\-:]+", cells[1]) or cells[0].lower() == "failure id":
            continue
        if not cells[0]:
            continue
        rows.append({
            "id": cells[0], "status": cells[1],
            "claimImpact": cells[2], "requiredResponse": cells[3] if len(cells) > 3 else "",
        })
    return rows


def _parse_sections(lines: list[str]) -> dict[str, str]:
    """Map ``## <id>`` section -> its authoritative ``**Status:** <STATUS>``."""
    sections: dict[str, str] = {}
    cur_id: str | None = None
    for line in lines:
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            cur_id = _norm(m.group(1))
            sections.setdefault(cur_id, "")
            continue
        if cur_id:
            sm = re.search(r"\*\*Status:?\*\*\s*(.+?)(?:\.|$)", line, re.IGNORECASE)
            if sm and not sections[cur_id]:
                sections[cur_id] = _norm(sm.group(1))
    return sections


def parse_ledger(path: Path = LEDGER) -> dict:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    table = _parse_table(lines)
    sections = _parse_sections(lines)
    return {"table": table, "sections": sections}


def validate(path: Path = LEDGER) -> dict:
    parsed = parse_ledger(path)
    table, sections = parsed["table"], parsed["sections"]
    missing: list[str] = []
    by_kind = {"open": 0, "resolved": 0, "other": 0}
    open_items: list[str] = []

    for row in table:
        rid, status, impact = row["id"], row["status"], row["claimImpact"]
        if not rid:
            missing.append(f"table row missing id: {row}")
        if not status:
            missing.append(f"{rid}: missing Status")
        if not impact:
            missing.append(f"{rid}: missing Claim impact")
        # section status overrides table status when a detailed write-up exists
        eff = sections.get(rid) or status
        kind = classify(eff)
        by_kind[kind] += 1
        if kind == "open":
            open_items.append(rid)

    # sections not in the table (detailed write-ups with no table row) count too
    table_ids = {r["id"] for r in table}
    for sid, status in sections.items():
        if sid in table_ids:
            continue
        kind = classify(status)
        by_kind[kind] += 1
        if kind == "open":
            open_items.append(sid)

    return {
        "ok": not missing and bool(table),
        "tableRows": len(table),
        "sections": len(sections),
        "byStatus": by_kind,
        "openCount": len(open_items),
        "openItems": sorted(set(open_items)),
        "missing": missing,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--check", action="store_true", help="exit 1 if structurally invalid")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = validate(args.ledger)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"failure ledger: {result['tableRows']} table rows, {result['sections']} detailed sections")
        print(f"  OPEN={result['byStatus']['open']}  RESOLVED={result['byStatus']['resolved']}  "
              f"OTHER={result['byStatus']['other']}")
        if result["openItems"]:
            print(f"  OPEN items ({len(result['openItems'])}):")
            for it in result["openItems"]:
                print(f"    - {it}")
        if result["missing"]:
            print(f"  STRUCTURAL PROBLEMS ({len(result['missing'])}):")
            for m in result["missing"]:
                print(f"    - {m}")
        print("  structural validity: " + ("OK" if result["ok"] else "INVALID"))
    if args.check:
        return 0 if result["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
