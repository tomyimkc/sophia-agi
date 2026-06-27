#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Disciplined corpus enrichment — grow the structured attribution corpus that bounds dataset
volume (M2 NO-GO: ~72 records -> ~880 rows; reaching 10k needs more RECORDS, not more rows/record).

The corpus is GROUND TRUTH for a SOURCE-DISCIPLINE model, so enrichment is fail-closed on accuracy:
every new record MUST carry a `provenance` note (the basis for the attribution) and pass schema +
sanity checks, and is auto-decontaminated against the eval/benchmark surfaces by the dataset build
(provenance_bench.dataset_guard EVAL_GLOBS already covers heldout_v1 + transfer_v1). This tool MERGES
a curated batch into data/attributions.json (de-duping by textId) and reports the volume it unlocks.

    python3 tools/enrich_corpus.py --batch data/corpus_enrichment/attributions_batch1.json --check
    python3 tools/enrich_corpus.py --batch data/corpus_enrichment/attributions_batch1.json --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTR = ROOT / "data" / "attributions.json"
ALLOWED_CONF = {"legendary", "compiled", "attributed", "anonymous", "contested", "none_extant", "pseudonymous"}
REQUIRED = {"textId", "domain", "recordType", "canonicalTitleEn", "tradition",
            "attributedAuthor", "authorConfidence", "doNotAttributeTo", "provenance"}


def validate(rec: dict, existing_keys: set) -> list[str]:
    errs = []
    miss = REQUIRED - set(rec)
    if miss:
        errs.append(f"missing fields {sorted(miss)}")
    if rec.get("authorConfidence") not in ALLOWED_CONF:
        errs.append(f"authorConfidence '{rec.get('authorConfidence')}' not in {sorted(ALLOWED_CONF)}")
    dn = rec.get("doNotAttributeTo") or []
    if not isinstance(dn, list) or not dn:
        errs.append("doNotAttributeTo must be a non-empty list (the false-author traps)")
    if rec.get("attributedAuthor") in dn:
        errs.append("attributedAuthor cannot also be in doNotAttributeTo")
    if not (rec.get("provenance") or "").strip():
        errs.append("provenance note is REQUIRED (auditability of the ground-truth fact)")
    if rec.get("textId") in existing_keys:
        errs.append(f"textId '{rec.get('textId')}' already in corpus (duplicate)")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=Path, required=True, help="JSON list of attribution records")
    ap.add_argument("--target", type=Path, default=ATTR)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="validate only, no write (default)")
    g.add_argument("--apply", action="store_true", help="merge the batch into the corpus")
    args = ap.parse_args()

    corpus = json.loads(args.target.read_text(encoding="utf-8"))
    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    if isinstance(batch, dict):
        batch = list(batch.values())
    existing = set(corpus)
    all_errs, ok = [], []
    seen_batch = set()
    for i, rec in enumerate(batch):
        errs = validate(rec, existing | seen_batch)
        if errs:
            all_errs.append((rec.get("textId", f"#{i}"), errs))
        else:
            ok.append(rec); seen_batch.add(rec["textId"])

    print(f"batch={len(batch)} valid={len(ok)} invalid={len(all_errs)} | corpus {len(corpus)} -> {len(corpus)+len(ok)}")
    for tid, errs in all_errs[:20]:
        print(f"  INVALID {tid}: {errs}")
    if all_errs:
        print("Fix the invalid records before --apply (corpus accuracy is fail-closed).")
        if not args.apply:
            return 1
    if args.apply:
        for rec in ok:
            corpus[rec["textId"]] = {k: v for k, v in rec.items()}
        args.target.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"APPLIED: corpus now {len(corpus)} records -> rebuild the dataset to realize the volume.")
        return 0
    print("CHECK-ONLY: pass --apply to merge.")
    return 0 if not all_errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
