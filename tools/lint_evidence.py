#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evidence linter — fail-closed if a record's claimed confidence exceeds what its evidence licenses.

Sibling to tools/lint_claims.py, but for PROVENANCE rather than prose. For each wiki page or
attribution record, this computes the MAX admissible authorConfidence from the record's typed,
independence-checked, recency-bounded evidence (okf.evidence_spec.derive_confidence over
evidence_spec.json) and FAILS if the claimed confidence is higher. This is a confidence-inflation
gate: N nominally-distinct sources that collapse to one origin, stale citations, single-origin
corroboration, and confidence-exceeding-evidence are all rejected.

A record supplies its evidence in one of two ways:
  * an explicit ``evidence`` list (each item: {type, confidence, sources:[{origin,observedDate}]}) —
    the precise form (used by the audit set); OR
  * frontmatter ``sources`` (list) — coarse fallback: each source is treated as one ``citation``
    evidence item so a page over-claiming past its lone citation is still caught.

Run:
    python3 tools/lint_evidence.py                     # lint wiki/ + data/attributions.json
    python3 tools/lint_evidence.py --jsonl PATH        # lint a JSONL of records (audit set form)
    python3 tools/lint_evidence.py --as-of 2026-07-01  # pin the recency reference date
Exit: 0 = clean, 1 = violations, 2 = unreadable inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okf import evidence_spec as es  # noqa: E402
from okf import schema  # noqa: E402
from okf.page import load_pages  # noqa: E402


def _record_from_page(page) -> dict:
    """Coerce an okf.Page into an evidence record. Falls back to treating each frontmatter
    source as a single 'citation' item when no explicit 'evidence' block is present."""
    meta = page.meta or {}
    claimed = meta.get("authorConfidence")
    evidence = meta.get("evidence")
    if not evidence:
        # Coarse fallback: each declared source is one citation with unknown (self) origin.
        srcs = schema.as_list(meta.get("sources"))
        evidence = [{"type": "citation", "confidence": claimed,
                     "sources": [{"id": str(s), "origin": None} for s in srcs]}] if srcs else []
    return {"id": page.id, "meta": meta, "evidence": evidence,
            "sources": schema.as_list(meta.get("sources")), "asOf": None}


def _record_from_attribution(text_id: str, rec: dict) -> dict:
    claimed = rec.get("authorConfidence")
    srcs = rec.get("sources") or []
    evidence = rec.get("evidence")
    if not evidence:
        # attributions.json records carry no source list today -> the record IS its own
        # single citation. This catches an attribution claiming 'consensus' from nothing.
        evidence = [{"type": "citation", "confidence": claimed,
                     "sources": ([{"id": str(s), "origin": None} for s in srcs]
                                 if srcs else [{"id": text_id, "origin": None}])}]
    return {"id": text_id, "meta": rec, "evidence": evidence, "sources": srcs, "asOf": None}


def evaluate_record(rec: dict, spec: dict, as_of: date | None = None) -> dict:
    """Core linter judgement for ONE record. Returns a dict with the verdict and reasons.

    verdict == 'reject' when the claimed authorConfidence exceeds the derived ceiling
    (confidence inflation), else 'accept'. This is the function the measure_* harness reuses,
    so the CLI and the measured false-admission rate share exactly one decision rule."""
    claimed = (rec.get("meta") or {}).get("authorConfidence")
    when = rec.get("asOf") or as_of
    d = es.derive_confidence(rec.get("evidence") or [], spec, claimed=claimed, as_of=when)
    verdict = "reject" if d.inflated else "accept"
    return {
        "id": rec.get("id"),
        "verdict": verdict,
        "claimed": claimed,
        "derived": d.derivedLabel,
        "claimedRank": d.claimedRank,
        "derivedRank": d.derivedRank,
        "effectiveIndependentCount": d.effectiveIndependentCount,
        "collapsed": d.collapsed,
        "reasons": d.reasons,
    }


def _iter_jsonl(path: Path):
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        yield json.loads(ln)


def lint(records: list[dict], spec: dict, as_of: date | None = None) -> dict:
    results = [evaluate_record(r, spec, as_of=as_of) for r in records]
    violations = [r for r in results if r["verdict"] == "reject"]
    return {
        "tool": "lint_evidence",
        "spec": "evidence_spec.json",
        "asOf": as_of.isoformat() if as_of else None,
        "recordsChecked": len(results),
        "violationCount": len(violations),
        "verdict": "FAIL" if violations else "OK",
        "violations": [
            {"id": v["id"], "claimed": v["claimed"], "derived": v["derived"],
             "reason": "claimed confidence '%s' exceeds evidence-licensed '%s': %s"
                       % (v["claimed"], v["derived"], "; ".join(v["reasons"]) or "inflation")}
            for v in violations
        ],
        "canClaimAGI": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="lint records from a JSONL file (audit-set form) instead of the wiki")
    ap.add_argument("--wiki", type=Path, default=ROOT / "wiki",
                    help="wiki directory to lint (default: wiki/)")
    ap.add_argument("--attributions", type=Path, default=ROOT / "data" / "attributions.json",
                    help="attributions.json to lint (default: data/attributions.json)")
    ap.add_argument("--spec", type=Path, default=None, help="evidence_spec.json path")
    ap.add_argument("--as-of", default=None, help="recency reference date (ISO-8601); "
                    "default per-record asOf or today")
    ap.add_argument("--no-wiki", action="store_true", help="skip the wiki scan")
    ap.add_argument("--no-attributions", action="store_true", help="skip the attributions scan")
    args = ap.parse_args()

    try:
        spec = es.load_spec(args.spec)
    except Exception as exc:
        print(json.dumps({"tool": "lint_evidence", "verdict": "ERROR",
                          "reason": f"unreadable spec: {exc}", "code": 2}))
        return 2

    as_of = None
    if args.as_of:
        try:
            as_of = date.fromisoformat(args.as_of)
        except ValueError:
            print(json.dumps({"tool": "lint_evidence", "verdict": "ERROR",
                              "reason": f"bad --as-of '{args.as_of}'", "code": 2}))
            return 2

    records: list[dict] = []
    try:
        if args.jsonl is not None:
            records = list(_iter_jsonl(args.jsonl))
        else:
            if not args.no_wiki and args.wiki.exists():
                for page in load_pages(args.wiki):
                    if (page.meta or {}).get("authorConfidence"):
                        records.append(_record_from_page(page))
            if not args.no_attributions and args.attributions.exists():
                attrs = json.loads(args.attributions.read_text(encoding="utf-8"))
                for tid, rec in attrs.items():
                    if isinstance(rec, dict) and rec.get("authorConfidence"):
                        records.append(_record_from_attribution(tid, rec))
    except Exception as exc:
        print(json.dumps({"tool": "lint_evidence", "verdict": "ERROR",
                          "reason": f"unreadable input: {exc}", "code": 2}))
        return 2

    receipt = lint(records, spec, as_of=as_of)

    if receipt["violationCount"]:
        print("EVIDENCE LINTER: FAIL — confidence inflation (claim exceeds evidence):",
              file=sys.stderr)
        for v in receipt["violations"]:
            print(f"  {v['id']}: {v['reason']}", file=sys.stderr)
    else:
        print(f"EVIDENCE LINTER: OK — {receipt['recordsChecked']} record(s), no inflation.",
              file=sys.stderr)
    print(json.dumps(receipt, ensure_ascii=False))
    return 1 if receipt["violationCount"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
