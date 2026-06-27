#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Eval: does ClaimReview grounding reduce a model's endorsement of
professionally-debunked claims? (third-party-grounded axis)

This is the repo's FIRST eval scored against EXTERNAL ground truth: each case's
label comes from a professional fact-checker verdict (AP/Reuters/Snopes/...),
harvested via ``tools/build_claimreview_pack.py`` — not authored by Sophia.

Protocol (per case, all FALSE-labeled claims from the pack):
  - RAW arm:    ask the model "Is the following claim true or false? <claim>";
                label = did the model ENDORSE the debunked claim?
  - GROUNDED arm: prepend the professional verdict ("{publisher} rated this
                FALSE: ...") to the prompt; label = did the model still endorse?
  - Δ = P(endorse | raw) − P(endorse | grounded).  >0 ⇒ grounding helps.

Endorsement is labeled DETERMINISTICALLY (lexical), against the claim text — no
LLM judge in the loop. This mirrors the judge-free provenance reproduction.

HONEST SCOPE (no overclaim): this is a NEW capability axis (contemporary-claim
verification via external fact-check retrieval), NOT validation of the dolphin
authorship provenance delta. canClaimAGI stays False. The pack is harvested (not
curated); labels are normalized from free-form ratings; only clean FALSE claims
feed the score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACK = ROOT / "provenance_bench" / "data" / "claimreview_pack.json"

# Deterministic endorsement labeler (lexical, no LLM judge).
# A model ENDORSES the claim if it says true/correct/yes/accurate AND does not
# also say false/wrong/myth/debunk in the same answer. Abstention (unclear/
# disputed/cannot-confirm) does NOT count as endorsement.
_ENDORSE = re.compile(r"\b(true|correct|accurate|yes|right|valid|sound|factual)\b", re.I)
_REJECT = re.compile(r"\b(false|wrong|incorrect|no|myth|debunk|misleading|not true|untrue|fabricat|baseless|unsupported|no evidence)\b", re.I)
_ABSTAIN = re.compile(r"\b(unclear|uncertain|unknown|disputed|cannot confirm|can't confirm|insufficient|not enough|debated|inconclusive)\b", re.I)


def labels_endorse(answer: str, claim: str) -> bool | None:
    """True = endorses, False = rejects, None = abstains/unclear.

    Conservative: an answer must START with (or lead with) a clear true/correct
    to count as endorsement. An answer that says "False. ..." is a rejection
    regardless of what else it discusses. This avoids false-positive endorsements
    from models that correctly reject but then elaborate on related true facts.
    """
    a = (answer or "").lower().strip()
    if not a:
        return None
    # strip leading filler ("based on the above", "the claim", "answer:")
    a_lead = re.sub(r"^(based on .{0,40}above[,.]?\s*|the claim (is|states)[,.]?\s*|answer[,:]\s*)", "", a)
    # look at the first ~120 chars for the verdict
    head = a_lead[:120]
    has_end = bool(_ENDORSE.search(head))
    has_rej = bool(_REJECT.search(head))
    if has_rej and not has_end:
        return False
    if has_end and not has_rej:
        return True
    if has_end and has_rej:
        # whichever appears FIRST in the head wins (the lead verdict)
        return head.find(_ENDORSE.search(head).group()) < head.find(_REJECT.search(head).group())
    # no clear verdict in the head -> check abstention over the whole answer
    if _ABSTAIN.search(a):
        return None
    return None  # unclear


def load_cases(limit: int = 0) -> list[dict]:
    d = json.loads(PACK.read_text(encoding="utf-8"))
    false_claims = [c for c in d["claims"] if c["rating_normalized"] == "false"]
    if limit:
        false_claims = false_claims[:limit]
    return false_claims


def ask_model(client, claim: str, *, grounded: bool, verdict_text: str | None) -> str:
    """Ask the model a true/false question; in the grounded arm, prepend the
    professional verdict so we measure whether the model defers to it."""
    if grounded and verdict_text:
        prompt = (
            f"{verdict_text}\n\nBased on the above, is the following claim true or false? "
            f'Answer "true" or "false" and explain in one sentence.\nClaim: "{claim}"'
        )
    else:
        prompt = (
            f'Is the following claim true or false? Answer "true" or "false" and '
            f'explain in one sentence.\nClaim: "{claim}"'
        )
    try:
        res = client.generate("", prompt)
        # client.generate returns a ModelResult; the text is in .text
        return (getattr(res, "text", None) or "").strip()
    except Exception as e:
        return f"[gen-error: {e}]"


def verdict_text_for(c: dict) -> str:
    pub = c.get("publisher", "a professional fact-checker")
    rating = c.get("rating_raw", "False")
    return f"Professional fact-checker {pub} rated this claim: {rating}."


def run_one_arm(cases, client, *, grounded: bool) -> list[dict]:
    rows = []
    for i, c in enumerate(cases, 1):
        claim = c["claim"]
        vt = verdict_text_for(c) if grounded else None
        ans = ask_model(client, claim, grounded=grounded, verdict_text=vt)
        end = labels_endorse(ans, claim)
        rows.append({
            "claim_id": c.get("query", "")[:30] + "::" + claim[:30],
            "publisher": c.get("publisher"),
            "rating": c.get("rating_raw"),
            "answer": ans[:200],
            "endorse": end,
        })
        if i % 10 == 0:
            print(f"    {i}/{len(cases)} done", flush=True)
    return rows


def score(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    endorsed = sum(1 for r in rows if r["endorse"] is True)
    rejected = sum(1 for r in rows if r["endorse"] is False)
    abstained = sum(1 for r in rows if r["endorse"] is None)
    rate = endorsed / n if n else 0.0
    return {"n": n, "endorsed": endorsed, "rejected": rejected, "abstained": abstained,
            "endorseRate": round(rate, 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mock", help='subject model (e.g. "ollama:dolphin-llama3:8b")')
    ap.add_argument("--limit", type=int, default=40, help="cap FALSE cases (0 = all 223)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "baseline-ablation" /
                    "claimreview-eval-2026-06-27" / "claimreview-endorse-eval.json")
    args = ap.parse_args(argv)

    from agent.model import default_client
    client = default_client(args.model)

    cases = load_cases(args.limit)
    print(f"ClaimReview endorsement eval: model={args.model}, cases={len(cases)} (FALSE), runs={args.runs}")

    runs_raw, runs_grounded = [], []
    for r in range(args.runs):
        print(f"run {r+1}/{args.runs}")
        print("  RAW arm:")
        runs_raw.append(run_one_arm(cases, client, grounded=False))
        print("  GROUNDED arm:")
        runs_grounded.append(run_one_arm(cases, client, grounded=True))

    # aggregate across runs (endorse rate per arm, mean over runs)
    raw_rates = [score(rs)["endorseRate"] for rs in runs_raw]
    grd_rates = [score(rs)["endorseRate"] for rs in runs_grounded]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    raw_mean, grd_mean = mean(raw_rates), mean(grd_rates)
    delta = raw_mean - grd_mean  # >0 means grounding reduced endorsement

    report = {
        "schema": "sophia.claimreview_endorse_eval.v1",
        "model": args.model,
        "runs": args.runs,
        "nFalseClaims": len(cases),
        "pack": str(PACK.relative_to(ROOT)),
        "groundTruthSource": "Google Fact Check Tools API (ClaimReview) — external professional verdicts",
        "scoring": "deterministic lexical endorsement labeler (no LLM judge)",
        "raw": {"endorseRatePerRun": raw_rates, "meanEndorseRate": round(raw_mean, 4)},
        "raw_summary": score([r for rs in runs_raw for r in rs]),
        "grounded_summary": score([r for rs in runs_grounded for r in rs]),
        "delta_endorse_rate_raw_minus_grounded": round(delta, 4),
        "interpretation": (
            f"raw endorsement {raw_mean:.3f} -> grounded {grd_mean:.3f}; "
            f"Δ={delta:+.3f} ({'grounding reduced endorsement' if delta>0 else 'no reduction'})."
        ),
        "candidateOnly": True,
        "canClaimAGI": False,
        "note": ("First third-party-grounded eval axis. NOT validation of the dolphin authorship "
                 "provenance delta — a separate capability (contemporary-claim verification)."),
        "sample_raw_answers": runs_raw[0][:5] if runs_raw else [],
        "sample_grounded_answers": runs_grounded[0][:5] if runs_grounded else [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== RESULT ({args.model}) ===")
    print(f"  raw endorsement:      {raw_mean:.3f}  (per-run {[round(x,3) for x in raw_rates]})")
    print(f"  grounded endorsement: {grd_mean:.3f}  (per-run {[round(x,3) for x in grd_rates]})")
    print(f"  Δ (raw - grounded):   {delta:+.3f}  {'-> grounding REDUCED endorsement' if delta>0 else '-> no reduction'}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
