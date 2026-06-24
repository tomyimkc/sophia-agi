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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seib", type=Path, default=SEIB)
    ap.add_argument("--split-out", type=Path, default=SPLIT_OUT)
    ap.add_argument("--examples-glob", default=EXAMPLES_GLOB)
    ap.add_argument("--audit", action="store_true", help="scan training examples for held-out leakage")
    ap.add_argument("--fail-on-leak", action="store_true", help="exit non-zero if any held-out entity leaks")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rows = load_contested(args.seib)
    split = split_contested(rows)
    print(f"contested={split['nContested']}  train={len(split['train'])}  heldout={len(split['heldout'])}")

    if not args.dry_run:
        args.split_out.parent.mkdir(parents=True, exist_ok=True)
        args.split_out.write_text(json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.split_out}")

    leak_count = 0
    if args.audit:
        heldout_entities = [tuple(e["entity"]) for e in split["heldout"]]
        examples = []
        for p in sorted(ROOT.glob(args.examples_glob)):
            try:
                examples.append((str(p.relative_to(ROOT)), json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError):
                continue
        findings = audit_leakage(heldout_entities, examples)
        leak_count = len(findings)
        if findings:
            print(f"\nLEAKAGE: {leak_count} training example(s) mention a HELD-OUT contested entity:")
            for f in findings[:50]:
                print(f"  - {f['file']}: {f['work']} / {f['author']}")
            print("These teach the held-out test. Move them to train-half entities or remove them.")
        else:
            print("\nLEAKAGE AUDIT: CLEAN — no training example covers a held-out contested entity.")

    if args.fail_on_leak and leak_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
