#!/usr/bin/env python3
"""C4 — close the continual loop: gate-feedback misses -> reviewed queue -> training rows.

The system already mines gate MISSES (judge caught a hallucination the gate let through)
into candidate ``doNotAttributeTo`` records (`agent/gate_feedback.py`) and learns provenance
rules from held-out failures (`provenance_bench/improvement.py`). What was missing is the
*return path*: turning those signals into the **next** training pack so the model improves,
not just the gate.

This tool provides that path in three explicit stages, with a human review gate so it stays
**non-circular** (the runtime gate never trains on its own unreviewed output):

  1. ``mine``      — read run/case results, emit candidate records to a pending queue
                     (deduped). Nothing is promoted; every candidate starts ``promoted:false``.
  2. ``approve``   — a reviewer flips a specific candidate to ``promoted:true`` with a note.
                     Default-deny: unreviewed candidates never advance.
  3. ``build-sft`` — convert ONLY promoted candidates into SFT rows
                     (`training/feedback/sft_from_feedback.jsonl`) + a promoted gate-records
                     file. `build_local_sophia_dataset.py` ingests the SFT file as a normal
                     source, so it passes the SAME decontamination guard (it cannot leak
                     eval/holdout prompts).

Non-circularity guarantees: pending candidates live in a separate file and are NEVER merged
into the frozen runtime records automatically; promotion requires an explicit human step;
ingested rows are decontaminated like any other source. Pure stdlib, deterministic, offline.
No weights change here; not an AGI claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate_feedback import candidate_record, detect_miss  # noqa: E402

FEEDBACK_DIR = ROOT / "training" / "feedback"
PENDING = FEEDBACK_DIR / "pending_candidates.jsonl"
SFT_OUT = FEEDBACK_DIR / "sft_from_feedback.jsonl"
PROMOTED_RECORDS = FEEDBACK_DIR / "promoted_records.jsonl"

SFT_SYSTEM = (
    "You are a precise instructor specializing in source discipline. You verify who actually "
    "authored texts versus traditional or mistaken attributions, and you refuse to confirm "
    "attributions that are not supported by evidence."
)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _candidate_key(c: dict) -> tuple:
    """Dedupe by (rid, forbidden markers) — the same identity gate_feedback uses."""
    return (c.get("rid"), tuple(sorted(c.get("doNotAttributeTo", []))))


def _candidate_tokens(c: dict) -> set:
    """Bag of lowercased word tokens for a candidate's identity (work + author + forbidden)."""
    import re

    parts = [str(c.get("work", "")), str(c.get("claimedAuthor", "")), *c.get("doNotAttributeTo", [])]
    text = " ".join(parts).lower()
    return {t for t in re.split(r"[^a-z0-9一-鿿]+", text) if t}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _novelty_ok(cand: dict, existing_token_sets: list[set], min_novelty: float) -> bool:
    """Diversity floor (Hurdle 4 — avoid self-data collapse).

    Exact-key dedup cannot stop the queue from *narrowing* onto near-duplicate misses,
    which is how self-improvement loops accumulate reward bias and lose answer diversity.
    A candidate passes only if its max token-Jaccard to anything already queued is
    < (1 - min_novelty). ``min_novelty=0.0`` disables the floor (back-compat default).
    """
    if min_novelty <= 0.0:
        return True
    tokens = _candidate_tokens(cand)
    sim_ceiling = 1.0 - min_novelty
    return all(_jaccard(tokens, prev) <= sim_ceiling for prev in existing_token_sets)


def make_candidate(work: str, claimed_author: str, *, mined_from: str) -> dict:
    """A reviewable, flat candidate carrying everything `build-sft` and review need."""
    rec = candidate_record(work, claimed_author)
    (rid, body), = rec.items()
    return {
        "rid": rid,
        "work": work,
        "claimedAuthor": claimed_author,
        "canonicalTitleEn": body["canonicalTitleEn"],
        "altTitlesEn": body["altTitlesEn"],
        "doNotAttributeTo": body["doNotAttributeTo"],
        "minedFrom": mined_from,
        "promoted": False,
        "reviewer": None,
        "note": None,
    }


def candidate_to_record(c: dict) -> dict:
    """Project a flat candidate back to a {rid: {...}} gate record."""
    return {c["rid"]: {
        "canonicalTitleEn": c["canonicalTitleEn"],
        "altTitlesEn": c.get("altTitlesEn", []),
        "doNotAttributeTo": c.get("doNotAttributeTo", []),
    }}


def candidate_to_sft(c: dict) -> dict:
    """Turn a promoted candidate into a source-discipline SFT example (messages format,
    same shape as training/corpus.jsonl). Trains the refuse-unsupported-attribution habit."""
    work = c["work"]
    claimed = c["claimedAuthor"]
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM},
            {"role": "user", "content": f"Did {claimed} write {work}?"},
            {"role": "assistant", "content": (
                f"No. {claimed} did not write {work}. This attribution is not supported by "
                f"evidence; treat it as a do-not-attribute case. Rely on verified provenance "
                f"rather than a popular, traditional, or assumed ascription, and say so plainly "
                f"when the authorship cannot be confirmed."
            )},
        ],
        "metadata": {
            "source": "gate-feedback",
            "project": "sophia-agi",
            "rid": c["rid"],
            "minedFrom": c.get("minedFrom"),
            "promoted": True,
            "reviewer": c.get("reviewer"),
            "note": c.get("note"),
            "notes": "Candidate mined from a gate MISS, human-reviewed, promoted to training.",
        },
    }


def cmd_mine(args: argparse.Namespace) -> int:
    """Mine candidates from run/case results into the pending queue (deduped)."""
    results = _read_jsonl(Path(args.case_results))
    pending = _read_jsonl(PENDING)
    seen = {_candidate_key(c) for c in pending}
    token_sets = [_candidate_tokens(c) for c in pending]
    min_novelty = float(getattr(args, "min_novelty", 0.0) or 0.0)
    added = 0
    skipped_low_novelty = 0
    for case in results:
        rec = detect_miss(case)
        if not rec:
            continue
        (rid, body), = rec.items()
        cand = make_candidate(body["canonicalTitleEn"], case.get("claimed_author", ""), mined_from=args.case_results)
        if _candidate_key(cand) in seen:
            continue
        if not _novelty_ok(cand, token_sets, min_novelty):
            skipped_low_novelty += 1
            continue
        pending.append(cand)
        seen.add(_candidate_key(cand))
        token_sets.append(_candidate_tokens(cand))
        added += 1
    _write_jsonl(PENDING, pending)
    print(json.dumps({"mined": added, "skippedLowNovelty": skipped_low_novelty,
                      "minNovelty": min_novelty, "pendingTotal": len(pending),
                      "promotedPending": sum(1 for c in pending if c.get("promoted")),
                      "queue": _rel(PENDING)}, indent=2))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    """Human review step: flip specific candidate(s) to promoted=true with a note."""
    pending = _read_jsonl(PENDING)
    rids = set(args.rid)
    touched = 0
    for c in pending:
        if c.get("rid") in rids:
            c["promoted"] = True
            c["reviewer"] = args.reviewer
            c["note"] = args.note
            touched += 1
    _write_jsonl(PENDING, pending)
    print(json.dumps({"approved": touched, "rids": sorted(rids),
                      "promotedTotal": sum(1 for c in pending if c.get("promoted"))}, indent=2))
    return 0 if touched else 1


def cmd_build_sft(args: argparse.Namespace) -> int:
    """Convert ONLY promoted candidates into SFT rows + a promoted gate-records file."""
    pending = _read_jsonl(PENDING)
    promoted = [c for c in pending if c.get("promoted") is True]
    sft_rows = [candidate_to_sft(c) for c in promoted]
    records = [candidate_to_record(c) for c in promoted]
    if not args.dry_run:
        _write_jsonl(SFT_OUT, sft_rows)
        _write_jsonl(PROMOTED_RECORDS, records)
    print(json.dumps({
        "promoted": len(promoted),
        "pendingUnreviewed": sum(1 for c in pending if not c.get("promoted")),
        "sftRows": len(sft_rows),
        "sftOut": _rel(SFT_OUT),
        "promotedRecordsOut": _rel(PROMOTED_RECORDS),
        "note": "Run tools/build_local_sophia_dataset.py to ingest (decontaminated) into the pack.",
    }, indent=2))
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    pending = _read_jsonl(PENDING)
    print(json.dumps({
        "pendingTotal": len(pending),
        "promoted": sum(1 for c in pending if c.get("promoted")),
        "unreviewed": sum(1 for c in pending if not c.get("promoted")),
        "sftRowsBuilt": len(_read_jsonl(SFT_OUT)),
    }, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="mine candidates from run/case results into the pending queue")
    m.add_argument("case_results", help="JSONL of run_case-style results")
    m.add_argument("--min-novelty", type=float, default=0.0,
                   help="diversity floor in [0,1): reject a candidate whose token-Jaccard to "
                        "anything already queued exceeds (1 - min_novelty). 0 disables (default).")
    m.set_defaults(func=cmd_mine)

    a = sub.add_parser("approve", help="human review: promote specific candidate rid(s)")
    a.add_argument("rid", nargs="+", help="record id(s) to promote")
    a.add_argument("--reviewer", default="unspecified")
    a.add_argument("--note", default=None)
    a.set_defaults(func=cmd_approve)

    b = sub.add_parser("build-sft", help="convert promoted candidates to SFT rows")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=cmd_build_sft)

    s = sub.add_parser("status", help="show queue counts")
    s.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
