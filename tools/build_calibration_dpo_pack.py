#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""W2/R2 — calibration records -> DPO preference pack (drop-in, fail-closed).

Bridges the last W2 gap between MEASURING calibration and TRAINING on it:
turns (prompt, answer, confidence, correct, action) records — the same shape
``agent.abstention_scoring`` scores — into DPO preference pairs a preference
trainer can consume, written as a pack ``tools/build_local_sophia_dataset.py``
picks up (decontaminated + manifest-tracked like every other pack).

Pair construction (BOTH directions, so this cannot train abstention-collapse):

  * confident-wrong  (action=answer, correct=False, confidence >= tau):
      rejected = the model's confident wrong answer
      chosen   = a calibrated abstention that names the uncertainty
  * answered-correct (action=answer, correct=True):
      chosen   = the model's correct answer   (KEEPS COVERAGE)
      rejected = an unnecessary abstention on an answerable prompt

FAIL-CLOSED balance guard: if either side is missing the tool refuses to emit
a pack — a one-sided pack is exactly the binary-scoring incentive problem the
abstention rubric exists to fix (guess-everything, or abstain-everything).

WHAT THIS DOES NOT DO (honest seam, pre-registered):
  * it does NOT run the preference-training step, and any adapter trained on
    this pack must (a) beat the POST-HOC Platt/temperature baseline from
    tools/train_calibration_objective.py on held-out ECE — not merely beat
    uncalibrated output — and (b) pass the post-training calibration/abstention
    re-audit (ECE, selective risk, answerable-coverage) before the promotion
    gate (tools/promote_adapter.py) may accept it. candidateOnly:true.

Record schema (JSONL):
  {"prompt": str, "answer": str, "confidence": float in [0,1],
   "correct": bool, "action": "answer"|"hedge"|"abstain"}
Usage:
  PYTHONPATH=. python3 tools/build_calibration_dpo_pack.py \
      --records recs.jsonl --out training/calibration_dpo.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from agent.calibration import expected_calibration_error
    _REPO_OK = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"

SCHEMA = "sophia.calibration_dpo_pack.v1"
DEFAULT_TAU = 0.7  # confident-wrong threshold; pre-registered, do not tune post-hoc

ABSTAIN_TEMPLATE = (
    "I don't have a verified basis to answer this confidently, so I won't assert "
    "an answer. My confidence would be too low to state this as fact; retrieving "
    "a source or narrowing the question would let me answer reliably."
)
NEEDLESS_ABSTAIN = (
    "I can't answer this."
)


def _env_artifact(reason: str) -> dict[str, Any]:
    return {"schema": SCHEMA, "ok": False, "reason": reason,
            "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False}


def build_pairs(records: "list[dict[str, Any]]", *, tau: float = DEFAULT_TAU) -> dict[str, Any]:
    """Build balanced DPO pairs from calibration records (fail-closed)."""
    if not (0.5 <= tau <= 1.0):
        return _env_artifact(f"tau must be in [0.5, 1.0]; got {tau}")
    confident_wrong, answered_correct, skipped = [], [], 0
    for r in records:
        prompt = str(r.get("prompt", "")).strip()
        answer = str(r.get("answer", "")).strip()
        if not prompt or not answer:
            skipped += 1  # no real texts -> nothing honest to train on
            continue
        try:
            conf = float(r.get("confidence"))
        except (TypeError, ValueError):
            skipped += 1
            continue
        action = str(r.get("action", "answer"))
        correct = bool(r.get("correct"))
        if action != "answer":
            continue  # hedges/abstains are not preference evidence either way
        if not correct and conf >= tau:
            confident_wrong.append({
                "prompt": prompt, "chosen": ABSTAIN_TEMPLATE, "rejected": answer,
                "meta": {"kind": "confident_wrong", "confidence": round(conf, 4)},
            })
        elif correct:
            answered_correct.append({
                "prompt": prompt, "chosen": answer, "rejected": NEEDLESS_ABSTAIN,
                "meta": {"kind": "answered_correct", "confidence": round(conf, 4)},
            })
    if not confident_wrong or not answered_correct:
        return _env_artifact(
            f"one-sided pack refused (confident_wrong={len(confident_wrong)}, "
            f"answered_correct={len(answered_correct)}): training on a single "
            "direction teaches guess-everything or abstain-everything (fail-closed)")
    pairs = confident_wrong + answered_correct
    ece_before = None
    if _REPO_OK:
        scored = [(float(r.get("confidence", 0.0)), bool(r.get("correct")))
                  for r in records
                  if str(r.get("action", "answer")) == "answer"
                  and isinstance(r.get("confidence"), (int, float))]
        if scored:
            ece_before = round(expected_calibration_error(
                [c for c, _ in scored], [k for _, k in scored]), 4)
    return {
        "schema": SCHEMA, "ok": True,
        "tau": tau, "nRecords": len(records), "nSkippedNoText": skipped,
        "pairs": pairs,
        "balance": {"confidentWrong": len(confident_wrong),
                    "answeredCorrect": len(answered_correct)},
        "eceBeforeOnRecords": ece_before,
        "preRegistered": {
            "baselineToBeat": "post-hoc Platt/temperature calibrator "
                              "(tools/train_calibration_objective.py) on held-out ECE",
            "reAudit": "post-training ECE + selective risk + answerable-coverage "
                       "before tools/promote_adapter.py",
        },
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W2/R2 calibration-records -> DPO pack")
    ap.add_argument("--records", required=True,
                    help="JSONL {prompt,answer,confidence,correct,action}")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help="confident-wrong confidence threshold (pre-registered default 0.7)")
    ap.add_argument("--out", default="training/calibration_dpo.jsonl",
                    help="pack path (picked up by build_local_sophia_dataset DPO_SOURCES)")
    ap.add_argument("--report", default=None, help="optional JSON report path")
    args = ap.parse_args(argv)

    if not _REPO_OK:
        print(json.dumps(_env_artifact(
            f"repo instruments unavailable ({_IMPORT_ERR}); run with PYTHONPATH=."), indent=2))
        return 2
    records = load_jsonl(Path(args.records))
    result = build_pairs(records, tau=args.tau)
    if result.get("ok"):
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for p in result["pairs"]:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"wrote {len(result['pairs'])} DPO pairs -> {out}")
    report = {k: v for k, v in result.items() if k != "pairs"}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
