#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Disciplined corpus enrichment — grow the structured corpora that bound dataset volume
(M2 NO-GO: ~72 records -> ~880 rows; reaching 10k needs more RECORDS, not more rows/record).

The corpus is GROUND TRUTH for a SOURCE-DISCIPLINE model, so enrichment is fail-closed on accuracy:
every new record MUST carry a `provenance` note and pass schema + sanity checks. New rows are
auto-decontaminated against the eval surfaces by the dataset build (EVAL_GLOBS covers heldout_v1 +
transfer_v1). Supports the record types whose generators drive volume: attributions, religion,
history.

    python3 tools/enrich_corpus.py --type attributions --batch <batch.json> --check
    python3 tools/enrich_corpus.py --type history      --batch <batch.json> --apply
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CONF = {"legendary", "compiled", "attributed", "anonymous", "contested", "none_extant",
                "pseudonymous", "disputed", "anachronism_risk", "established"}

# type -> (target file, key field, required fields, list field that must be non-empty)
TYPES = {
    "attributions": ("data/attributions.json", "textId",
                     {"textId", "domain", "recordType", "canonicalTitleEn", "tradition",
                      "attributedAuthor", "authorConfidence", "doNotAttributeTo", "provenance"},
                     "doNotAttributeTo"),
    "religion": ("data/religion_concepts.json", "recordId",
                 {"recordId", "domain", "recordType", "canonicalTitleEn", "tradition",
                  "doNotMergeWith", "provenance"}, "doNotMergeWith"),
    "history": ("data/history_events.json", "recordId",
                {"recordId", "domain", "recordType", "canonicalTitleEn", "region",
                 "dateConsensus", "authorConfidence", "provenance"}, None),
}


def validate(rec: dict, spec, existing: set) -> list[str]:
    _file, key, required, listfield = spec
    errs = []
    miss = required - set(rec)
    if miss:
        errs.append(f"missing fields {sorted(miss)}")
    if rec.get("authorConfidence") and rec["authorConfidence"] not in ALLOWED_CONF:
        errs.append(f"authorConfidence '{rec.get('authorConfidence')}' not in {sorted(ALLOWED_CONF)}")
    if listfield:
        lv = rec.get(listfield) or []
        if not isinstance(lv, list) or not lv:
            errs.append(f"{listfield} must be a non-empty list")
        if rec.get("attributedAuthor") and rec.get("attributedAuthor") in lv:
            errs.append("attributedAuthor cannot also be in the forbidden list")
    if not (rec.get("provenance") or "").strip():
        errs.append("provenance note REQUIRED (auditability of the ground-truth fact)")
    if rec.get(key) in existing:
        errs.append(f"{key} '{rec.get(key)}' already in corpus (duplicate)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", choices=list(TYPES), default="attributions")
    ap.add_argument("--batch", type=Path, required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    spec = TYPES[args.type]
    target = ROOT / spec[0]
    key = spec[1]
    corpus = json.loads(target.read_text(encoding="utf-8"))
    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    if isinstance(batch, dict):
        batch = list(batch.values())
    existing = set(corpus)
    ok, bad, seen = [], [], set()
    for i, rec in enumerate(batch):
        errs = validate(rec, spec, existing | seen)
        if errs:
            bad.append((rec.get(key, f"#{i}"), errs))
        else:
            ok.append(rec); seen.add(rec[key])

    print(f"[{args.type}] batch={len(batch)} valid={len(ok)} invalid={len(bad)} | "
          f"corpus {len(corpus)} -> {len(corpus)+len(ok)}")
    for tid, errs in bad[:25]:
        print(f"  INVALID {tid}: {errs}")
    if args.apply:
        if bad:
            print("Refusing to apply: fix invalid records first (accuracy is fail-closed)."); return 1
        for rec in ok:
            corpus[rec[key]] = dict(rec)
        target.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"APPLIED: {spec[0]} now {len(corpus)} records -> rebuild the dataset.")
        return 0
    print("CHECK-ONLY: pass --apply to merge." if not bad else "Fix INVALID records before --apply.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
