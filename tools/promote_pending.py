#!/usr/bin/env python3
"""Promote verified gate-miss candidates into the live record set — the second half
of the active-learning loop.

``agent.gate_feedback`` logs gate misses (a forbidden attribution the judge caught
but the gate let through) to a pending JSONL queue. This tool closes the loop:

  1. read the pending queue,
  2. RE-VERIFY each candidate against independent ground truth — the documented
     true author (offline Wikidata snapshot / grounded resolver) must confirm the
     claimed author is genuinely wrong (NOT a pen name / variant — reuses the same
     conservative disambiguation as the grounded gate),
  3. drop anything already covered by a live record (dedupe),
  4. on ``--apply``, merge the survivors into ``data/learned_attributions.json``,
     which the live gate reads (``_PROVENANCE_FILES``), so the next run catches them.

Default is a DRY RUN (prints what would be promoted). Promotion never edits the
seed domain files; the learned sink is separate and auditable. Offline.

    python tools/promote_pending.py                 # dry run (report)
    python tools/promote_pending.py --apply         # write survivors to the live sink
    python tools/promote_pending.py --pending PATH --sink PATH
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PENDING_DEFAULT = ROOT / "agi-proof" / "benchmark-results" / "gate-pending-records.jsonl"
SINK_DEFAULT = ROOT / "data" / "learned_attributions.json"


def _norm(s: str) -> str:
    import re

    s = re.sub(r"\s*\(.*?\)\s*", " ", s or "")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'’-]", " ", s)).strip().lower()


def _load_pending(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _live_record_keys() -> set:
    """(rid, frozenset(forbidden markers)) for every live record, to dedupe against."""
    from agent.verifiers import _load_provenance_records

    keys = set()
    for rid, rec in _load_provenance_records().items():
        markers = frozenset(_norm(m) for m in rec.get("doNotAttributeTo", []))
        keys.add((rid, markers))
    return keys


def _reverify(work: str, claimed_author: str) -> "tuple[bool, str]":
    """Independently confirm the claimed author is genuinely wrong for the work.

    Reuses agent.grounded_gate: resolve the documented true author and apply the
    SAME conservative same-person / recognized-distinct-author guards used by the
    live grounded gate, so a correct pen name / variant is NOT promoted. Returns
    (confirmed_wrong, reason)."""
    try:
        from agent.grounded_gate import synth_records_for_claim
    except Exception as exc:  # pragma: no cover
        return (False, f"grounded gate unavailable: {exc}")
    claim = f"{claimed_author} wrote {work}."
    synthesized = synth_records_for_claim(claim, base_records={})
    if synthesized:
        # grounded gate confirmed a different documented author -> genuinely wrong.
        return (True, "confirmed by grounded resolver (documented author differs)")
    return (False, "could not independently confirm a different true author")


def promote(pending_path: Path, sink_path: Path, *, apply: bool) -> dict:
    pending = _load_pending(pending_path)
    live_keys = _live_record_keys()

    sink = {}
    if sink_path.exists():
        try:
            sink = json.loads(sink_path.read_text(encoding="utf-8")) or {}
        except json.JSONDecodeError:
            sink = {}
    sink_keys = {
        (rid, frozenset(_norm(m) for m in rec.get("doNotAttributeTo", [])))
        for rid, rec in sink.items()
    }

    promoted, skipped_dupe, skipped_unconfirmed = [], [], []
    seen_this_run = set()

    for candidate in pending:
        if not isinstance(candidate, dict) or len(candidate) != 1:
            continue
        (topkey, rec), = candidate.items()
        # Match how _merge_records keys live records (recordId > textId > topkey), so
        # dedup is consistent even if a candidate carries an explicit recordId/textId.
        rid = rec.get("recordId") or rec.get("textId") or topkey
        markers = frozenset(_norm(m) for m in rec.get("doNotAttributeTo", []))
        key = (rid, markers)
        if key in live_keys or key in sink_keys or key in seen_this_run:
            skipped_dupe.append(rid)
            continue
        work = rec.get("canonicalTitleEn", rid.replace("_", " "))
        claimed = (rec.get("doNotAttributeTo") or [""])[0]
        ok, reason = _reverify(work, claimed)
        if not ok:
            skipped_unconfirmed.append({"rid": rid, "reason": reason})
            continue
        seen_this_run.add(key)
        promoted.append({"rid": rid, "record": rec, "reason": reason})

    if apply and promoted:
        for p in promoted:
            sink[p["rid"]] = p["record"]
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        sink_path.write_text(json.dumps(sink, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "pending": len(pending),
        "promoted": promoted,
        "skippedDuplicate": skipped_dupe,
        "skippedUnconfirmed": skipped_unconfirmed,
        "applied": bool(apply and promoted),
        "sink": str(sink_path.relative_to(ROOT)) if sink_path.is_relative_to(ROOT) else str(sink_path),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pending", type=Path, default=PENDING_DEFAULT)
    ap.add_argument("--sink", type=Path, default=SINK_DEFAULT)
    ap.add_argument("--apply", action="store_true", help="write survivors to the live learned sink (default: dry run)")
    args = ap.parse_args(argv)

    result = promote(args.pending, args.sink, apply=args.apply)
    print(f"pending={result['pending']} "
          f"promoted={len(result['promoted'])} "
          f"dupe={len(result['skippedDuplicate'])} "
          f"unconfirmed={len(result['skippedUnconfirmed'])} "
          f"applied={result['applied']}")
    for p in result["promoted"]:
        print(f"  + {p['rid']}: do-not-attribute {p['record'].get('doNotAttributeTo')} ({p['reason']})")
    for s in result["skippedUnconfirmed"]:
        print(f"  - skip {s['rid']}: {s['reason']}")
    if result["promoted"] and not args.apply:
        print("\n(dry run — re-run with --apply to write these into the live gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
