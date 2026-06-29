#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the prompt-quality gate on a held-out pack — lint -> validated gate (Layer C).

`agent/prompt_quality_verifier.py` is the metric. On its own it is a *lint*: a few hand-written
fixtures prove the wiring, not that it generalises. This harness runs the predicate
(`prompt_quality_ok`) over a held-out, labelled pack it never saw and reports precision / recall /
accuracy against a promotion floor — the same held-out discipline `gateway.skill_flywheel.
synthesize_gate` enforces before a forged gate is promoted. Only if the predicate clears the floor
on data it did not author is it a *validated gate* worth gating generated prompts with.

Pack: ``eval/prompt_quality/heldout_v1.jsonl`` — rows ``{"id", "label": bool, "text"}`` where
``label`` is the human judgement "is this a well-formed task/PR/handoff prompt?". Deterministic,
offline, no model, no network.

    python tools/eval_prompt_quality.py
    python tools/eval_prompt_quality.py --emit agi-proof/benchmark-results/prompt-quality-heldout.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prompt_quality_verifier import prompt_quality_ok  # noqa: E402

PACK = ROOT / "eval" / "prompt_quality" / "heldout_v1.jsonl"
MIN_PRECISION = 0.8
MIN_RECALL = 0.8


def _load(path: Path = PACK) -> "list[dict]":
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate(rows: "list[dict] | None" = None) -> dict:
    rows = rows if rows is not None else _load()
    tp = fp = tn = fn = 0
    errors: list[dict] = []
    for r in rows:
        gold = bool(r["label"])
        pred = prompt_quality_ok(r["text"])
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
            errors.append({"id": r.get("id"), "type": "false_accept", "text": r["text"][:80]})
        elif not pred and gold:
            fn += 1
            errors.append({"id": r.get("id"), "type": "false_reject", "text": r["text"][:80]})
        else:
            tn += 1
    n = len(rows)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    accuracy = (tp + tn) / n if n else 0.0
    promoted = precision >= MIN_PRECISION and recall >= MIN_RECALL
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4), "recall": round(recall, 4), "accuracy": round(accuracy, 4),
        "minPrecision": MIN_PRECISION, "minRecall": MIN_RECALL,
        "promoted": promoted,
        "verdict": "FLOOR-MET" if promoted else "FLOOR-UNMET",
        "caveat": ("heldout_v1 is the DEVELOPMENT pack the detectors were tuned against; "
                   "floor-met here shows self-consistency, NOT generalization. An independent, "
                   "un-tuned pack (v2) is the real held-out gate and is OPEN — do not call this a "
                   "validated gate until v2 clears."),
        "errors": errors,
    }


def offline_invariants() -> "tuple[bool, dict]":
    res = evaluate()
    checks = {
        "pack_nonempty": res["n"] >= 20,
        "has_both_labels": (res["tp"] + res["fn"]) >= 1 and (res["tn"] + res["fp"]) >= 1,
        "precision_floor_met": res["precision"] >= MIN_PRECISION,
        "recall_floor_met": res["recall"] >= MIN_RECALL,
        "separates_dev_pack": res["promoted"] is True,  # dev-set self-consistency, NOT generalization
    }
    return all(checks.values()), {"checks": checks, "result": res}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", type=Path, help="write the result JSON here")
    args = ap.parse_args(argv)
    res = evaluate()
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}, ensure_ascii=False, indent=2))
    if res["errors"]:
        print("misclassified:", json.dumps(res["errors"], ensure_ascii=False))
    if args.emit:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote -> {args.emit}")
    return 0 if res["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
