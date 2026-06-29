#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate a discipline verifier on an independent labelled pack (recall on bad, pass-rate on good).

A discipline verifier (e.g. ``finance_sound``, ``medicine_safe``) is only trustworthy if it
SEPARATES a fresh labelled set it did not author its self-checks from: it should REJECT the bad
answers (high recall) and ACCEPT the good ones (low false-positive). This runs the RAW domain
verifier (not the composite council gate, to isolate the new verifier) over a v2 pack and reports
recall / pass-rate against a floor.

Honest scope (pre-registered, cf. the prompt-quality gate): the v2 packs are independent of the
verifiers' self-checks and of ``heldout_v1``, but they were authored knowing the verifier's rules —
so floor-met here shows the verifier is self-consistent on a fresh set, NOT that it generalises to a
truly blind third-party pack. That remains OPEN.

    python tools/eval_discipline_verifier.py finance eval/council/finance_heldout_v2.jsonl
    python tools/eval_discipline_verifier.py medicine eval/council/medicine_heldout_v2.jsonl --emit out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIN_RECALL = 0.9
MIN_PASS = 0.9


def _verifier(discipline: str):
    if discipline == "finance":
        from agent.finance_verifier import finance_sound
        return finance_sound()
    if discipline == "medicine":
        from agent.medicine_verifier import medicine_safe
        return medicine_safe()
    if discipline == "chemistry":
        from agent.chemistry_verifier import chemistry_sound
        return chemistry_sound()
    if discipline == "biology":
        from agent.biology_verifier import biology_sound
        return biology_sound()
    raise SystemExit(f"no standalone verifier registered for discipline '{discipline}'")


def _load(path: Path) -> "list[dict]":
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(discipline: str, rows: "list[dict]") -> dict:
    v = _verifier(discipline)
    n_bad = sum(1 for r in rows if not r["correct"])
    n_good = sum(1 for r in rows if r["correct"])
    caught = passed_good = 0
    errors = []
    for r in rows:
        ok = v(r["answer"])["passed"]
        if not r["correct"]:
            caught += int(not ok)
            if ok:
                errors.append({"id": r.get("id"), "type": "false_accept", "answer": r["answer"][:70]})
        else:
            passed_good += int(ok)
            if not ok:
                errors.append({"id": r.get("id"), "type": "false_reject", "answer": r["answer"][:70]})
    recall = caught / n_bad if n_bad else 1.0
    pass_good = passed_good / n_good if n_good else 1.0
    promoted = recall >= MIN_RECALL and pass_good >= MIN_PASS
    return {
        "discipline": discipline, "n": len(rows), "nBad": n_bad, "nGood": n_good,
        "recallOnBad": round(recall, 4), "passRateOnGood": round(pass_good, 4),
        "minRecall": MIN_RECALL, "minPass": MIN_PASS, "floorMet": promoted,
        "caveat": ("v2 is independent of the self-checks and heldout_v1 but authored knowing the "
                   "verifier's rules -> self-consistency on a fresh set, NOT blind generalisation; "
                   "a truly third-party pack remains OPEN."),
        "errors": errors,
    }


def offline_invariants() -> "tuple[bool, dict]":
    fin = evaluate("finance", _load(ROOT / "eval" / "council" / "finance_heldout_v2.jsonl"))
    med = evaluate("medicine", _load(ROOT / "eval" / "council" / "medicine_heldout_v2.jsonl"))
    checks = {
        "finance_floor_met": fin["floorMet"],
        "medicine_floor_met": med["floorMet"],
        "finance_has_both_labels": fin["nBad"] >= 3 and fin["nGood"] >= 3,
        "medicine_has_both_labels": med["nBad"] >= 3 and med["nGood"] >= 3,
    }
    return all(checks.values()), {"checks": checks, "finance": fin, "medicine": med}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("discipline")
    ap.add_argument("pack", type=Path)
    ap.add_argument("--emit", type=Path)
    args = ap.parse_args(argv)
    r = evaluate(args.discipline, _load(args.pack))
    print(json.dumps({k: v for k, v in r.items() if k != "errors"}, ensure_ascii=False, indent=2))
    if r["errors"]:
        print("misclassified:", json.dumps(r["errors"], ensure_ascii=False))
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if r["floorMet"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
