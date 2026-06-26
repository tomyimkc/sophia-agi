#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SEIB contested-entity generalization split + teaching-to-the-test leakage audit.

Why this exists: fixing `fabricationRateOnContested == 0` by authoring a qualification
trace for each failing SEIB-100 row (Beauvoir, Bradbury, Dostoevsky, ...) is
teaching-to-the-test at the ENTITY level. The verbatim-shingle decontam guard misses it
(the prompt wording differs, but the (work, gold_author) pair is identical), so a 0.0
reached that way is memorization, not a habit — and it will not survive a third-party
SEIB pack (Hurdle 1).

The honest control: split the SEIB-100 contested entities deterministically into a TRAIN
half and a HELD-OUT half. Instil the qualification habit using ONLY train-half (or fully
disjoint) entities, then require fabricationRateOnContested == 0 on the HELD-OUT half —
entities the adapter was never trained to qualify. Generalisation there is evidence of a
habit; 0.0 only on trained entities is memorisation.

  python3 tools/seib_generalization_split.py                 # write the split
  python3 tools/seib_generalization_split.py --audit         # + flag training files that
                                                             #   leak a HELD-OUT entity
  python3 tools/seib_generalization_split.py --audit --fail-on-leak   # CI gate

This is bookkeeping only — no model, no training, deterministic (hash-based, no RNG).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEIB = ROOT / "eval" / "seib" / "seib_100_v1.jsonl"
SPLIT_OUT = ROOT / "eval" / "seib" / "seib_contested_split.json"
EXAMPLES_GLOB = "training/examples/*.json"


def load_contested(path: Path = SEIB) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("label") == "qualify_or_abstain"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text).lower())).strip()


def author_core(gold_author: str | None) -> str:
    """Core author name: text before any parenthetical, normalised ('Confucius (compiled
    by his disciples)' -> 'confucius'). Empty if no documented gold author."""
    if not gold_author:
        return ""
    return _norm(gold_author.split("(")[0])


def entity_of(row: dict) -> tuple[str, str]:
    return (_norm(row.get("work", "")), author_core(row.get("gold_author")))


def _bucket(row_id: str) -> str:
    """Deterministic, RNG-free 50/50 assignment by SHA1 parity of the case id."""
    h = int(hashlib.sha1(row_id.encode("utf-8")).hexdigest(), 16)
    return "train" if h % 2 == 0 else "heldout"


def split_contested(rows: list[dict]) -> dict[str, Any]:
    train, heldout = [], []
    for r in rows:
        (train if _bucket(str(r["id"])) == "train" else heldout).append({
            "id": r["id"], "work": r.get("work"), "gold_author": r.get("gold_author"),
            "entity": list(entity_of(r)),
        })
    return {
        "schema": "sophia.seib_contested_split.v1",
        "source": str(SEIB.relative_to(ROOT)),
        "nContested": len(rows),
        "train": train,
        "heldout": heldout,
        "protocol": ("Train the qualification habit on TRAIN (or fully disjoint) entities only; "
                     "require fabricationRateOnContested==0 on HELD-OUT entities never trained. "
                     "0.0 on held-out = habit; 0.0 only on trained = memorisation (teaching-to-test)."),
    }


def example_text(obj: dict) -> str:
    parts = [m.get("content", "") for m in obj.get("messages", []) if isinstance(m, dict)]
    return _norm(" ".join(parts) + " " + json.dumps(obj.get("metadata", {}), ensure_ascii=False))


def audit_leakage(heldout_entities: list[tuple[str, str]], examples: list[tuple[str, dict]]) -> list[dict]:
    """Flag a training example that mentions BOTH the work and the gold author of a
    HELD-OUT contested entity — i.e. it would teach the held-out test case."""
    findings = []
    for name, obj in examples:
        text = example_text(obj)
        for work, author in heldout_entities:
            if work and author and work in text and author in text:
                findings.append({"file": name, "work": work, "author": author})
    return findings


def corpus_partition(contested_rows: list[dict], examples: list[tuple[str, dict]]) -> dict[str, Any]:
    """Partition contested entities by whether the ACTIVE corpus already teaches them.

    Distinguishes the legitimate mission (the corpus teaching provenance on canonical works
    like the Analects) from a clean generalization held-out. Only ``corpusClean`` entities —
    those NO training example covers — can measure whether the qualification habit TRANSFERS
    to unseen entities. ``corpusTaught`` entities measure in-distribution retention, not
    generalization.
    """
    texts = [(name, example_text(obj)) for name, obj in examples]
    clean, taught = [], []
    for row in contested_rows:
        work, author = entity_of(row)
        n = 0
        if work and author:
            n = sum(1 for _name, t in texts if work in t and author in t)
        rec = {"id": row["id"], "work": row.get("work"), "gold_author": row.get("gold_author"),
               "entity": [work, author], "corpusExamples": n}
        (taught if n > 0 else clean).append(rec)
    return {
        "schema": "sophia.seib_contested_corpus_partition.v1",
        "nContested": len(contested_rows),
        "corpusClean": clean,
        "corpusTaught": taught,
        "nCorpusClean": len(clean),
        "nCorpusTaught": len(taught),
        "guidance": ("Use corpusClean entities as the GENERALIZATION held-out (fabrication==0 there "
                     "= a transferring habit). corpusTaught entities measure in-distribution retention "
                     "only. If nCorpusClean is too small for a stable rate, commission a fresh "
                     "corpus-disjoint contested pack (this also advances Hurdle 1 / external validation)."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seib", type=Path, default=SEIB)
    ap.add_argument("--split-out", type=Path, default=SPLIT_OUT)
    ap.add_argument("--examples-glob", default=EXAMPLES_GLOB)
    ap.add_argument("--audit", action="store_true", help="scan training examples for held-out leakage")
    ap.add_argument("--fail-on-leak", action="store_true", help="exit non-zero if any held-out entity leaks")
    ap.add_argument("--partition-by-corpus", action="store_true",
                    help="partition contested entities into corpus-clean (generalization held-out) "
                         "vs corpus-taught (in-distribution), by actual corpus coverage")
    ap.add_argument("--partition-out", type=Path,
                    default=ROOT / "eval" / "seib" / "seib_contested_corpus_partition.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rows = load_contested(args.seib)
    split = split_contested(rows)
    print(f"contested={split['nContested']}  train={len(split['train'])}  heldout={len(split['heldout'])}")

    if not args.dry_run:
        args.split_out.parent.mkdir(parents=True, exist_ok=True)
        args.split_out.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.split_out}")

    examples: list[tuple[str, dict]] = []
    if args.audit or args.partition_by_corpus:
        for p in sorted(ROOT.glob(args.examples_glob)):
            try:
                examples.append((str(p.relative_to(ROOT)), json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                continue

    if args.partition_by_corpus:
        part = corpus_partition(rows, examples)
        print(f"\ncorpus partition: corpusClean={part['nCorpusClean']} "
              f"(generalization held-out)  corpusTaught={part['nCorpusTaught']} (in-distribution)")
        for e in part["corpusClean"][:50]:
            print(f"  clean: {e['work']} / {e['gold_author']}")
        if not args.dry_run:
            args.partition_out.write_text(json.dumps(part, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {args.partition_out}")
        if part["nCorpusClean"] < 10:
            print("WARNING: too few corpus-clean contested entities for a stable rate — "
                  "commission a fresh corpus-disjoint pack (Hurdle 1).")

    leak_count = 0
    if args.audit:
        # Prefer the corpus-clean set as the held-out when the partition was computed — that
        # is the CORRECT generalization held-out (the naive hash split can misclassify the few
        # corpus-taught entities). Auditing the clean set guards against NEW traces leaking it.
        if args.partition_by_corpus:
            heldout_entities = [tuple(e["entity"]) for e in corpus_partition(rows, examples)["corpusClean"]]
            held_label = "corpus-clean generalization"
        else:
            heldout_entities = [tuple(e["entity"]) for e in split["heldout"]]
            held_label = "hash-split held-out"
        findings = audit_leakage(heldout_entities, examples)
        leak_count = len(findings)
        if findings:
            print(f"\nLEAKAGE ({held_label}): {leak_count} training example(s) teach a held-out contested entity:")
            for f in findings[:50]:
                print(f"  - {f['file']}: {f['work']} / {f['author']}")
            print("Remove these or move the qualification habit to non-held-out entities.")
        else:
            print(f"\nLEAKAGE AUDIT ({held_label}): CLEAN — no training example covers a held-out contested entity.")

    if args.fail_on_leak and leak_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
