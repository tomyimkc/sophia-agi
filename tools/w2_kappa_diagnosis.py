#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Diagnose the W2 scorer-vs-Claude kappa gap.

Re-derives the per-answer 'fabricated?' label from BOTH the deterministic scorer and the
Claude judge (via the llmhub relay), aligned on the same 108 abstain-pack answers (3 seeds
x 3 modes x 12 abstain cases), and characterizes WHERE they disagree so the kappa=-0.03
gap is explainable rather than a black box.

Outputs a per-case table + a disagreement-pattern summary to
agi-proof/baseline-ablation/w2-kappa-disagreement-2026-06-26.json

Needs OPENAI_API_KEY (the llmhub relay key) + the base_url override. Offline scorer only.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.calibration_judge import judge_answer  # noqa: E402
from provenance_bench.calibration_score import score_answer  # noqa: E402

MODES = ("sophia-full", "raw-model", "raw-model-plus-tools")
PACK = ROOT / "agi-proof" / "baseline-ablation" / "abstain-pack-2026-06-22.json"
DUMPS = sorted((ROOT / "agi-proof" / "baseline-ablation").glob("w2-ablation-2026-06-26.seed*.private.json"))
JUDGE_SPEC = os.environ.get("W2_JUDGE_SPEC", "openai:claude-sonnet-4-6@https://api.llmhub.com.cn/v1")
OUT = ROOT / "agi-proof" / "baseline-ablation" / "w2-kappa-disagreement-2026-06-26.json"


def _judge_fn(prompt: str) -> str:
    from agent.model import complete
    # complete returns the assistant text directly on this version
    return complete("You are a strict evaluator.", prompt, spec=JUDGE_SPEC, max_tokens=24)


def main() -> int:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    abstain = [c for c in pack["cases"] if c.get("epistemicLabel") == "abstain"]
    print(f"abstain cases: {len(abstain)} | dumps: {len(DUMPS)} | judge: {JUDGE_SPEC}")

    rows = []
    for di, dump_path in enumerate(DUMPS):
        dump = json.loads(dump_path.read_text(encoding="utf-8"))
        for mode in MODES:
            responses = (dump.get(mode) or {}).get("responses", {})
            if not responses:
                continue
            for case in abstain:
                ans = responses.get(case["id"], "")
                if not ans.strip():
                    continue
                s = score_answer(ans, case)
                scorer_fab = bool(s["fabricated"])
                claude_fab = judge_answer(case.get("prompt", ""), ans, judge_fn=_judge_fn) == "fabricated"
                agree = scorer_fab == claude_fab
                rows.append({
                    "caseId": case["id"],
                    "seed": di,
                    "mode": mode,
                    "scorerFabricated": scorer_fab,
                    "claudeFabricated": claude_fab,
                    "agree": agree,
                    "scorerState": s.get("state"),
                    "answerLen": len(ans),
                    "answerHead": ans[:160].replace("\n", " "),
                })

    n = len(rows)
    a = sum(1 for r in rows if r["agree"])
    print(f"\npaired answers: {n} | agree: {a} ({a/n:.1%}) | disagree: {n-a}")

    # disagreement breakdown by direction + mode
    def cat(r):
        if r["scorerFabricated"] and not r["claudeFabricated"]:
            return "scorer_says_fab_claude_says_not"  # scorer stricter
        if r["claudeFabricated"] and not r["scorerFabricated"]:
            return "claude_says_fab_scorer_says_not"  # claude stricter
        return None
    for r in rows:
        r["disagreeCategory"] = cat(r)

    print("\n=== disagreement by direction ===")
    dirc = Counter(r["disagreeCategory"] for r in rows if r["disagreeCategory"])
    for k, v in dirc.most_common():
        print(f"  {k}: {v}")

    print("\n=== disagreement by mode (which pipeline arm) ===")
    for mode in MODES:
        dis = [r for r in rows if r["mode"] == mode and not r["agree"]]
        tot = [r for r in rows if r["mode"] == mode]
        if tot:
            print(f"  {mode}: {len(dis)}/{len(tot)} disagree ({len(dis)/len(tot):.1%})")

    print("\n=== scorer-state of disagreements (what the scorer called them) ===")
    stc = Counter(r["scorerState"] for r in rows if not r["agree"])
    for k, v in stc.most_common():
        print(f"  {k}: {v}")

    # sample disagreements for human inspection
    samples = {}
    for label in ("scorer_says_fab_claude_says_not", "claude_says_fab_scorer_says_not"):
        ds = [r for r in rows if r["disagreeCategory"] == label][:6]
        samples[label] = [{"caseId": r["caseId"], "mode": r["mode"], "seed": r["seed"],
                           "scorerState": r["scorerState"], "answerHead": r["answerHead"]} for r in ds]

    report = {
        "schema": "sophia.w2_kappa_disagreement.v1",
        "candidateOnly": True, "validated": False, "level3Evidence": False, "canClaimAGI": False,
        "question": "Why is scorer-vs-Claude kappa ~ -0.03 on the W2 abstain pack (bar 0.40)?",
        "judgeSpec": JUDGE_SPEC,
        "nPaired": n, "nAgree": a, "nDisagree": n - a,
        "disagreementByDirection": dict(dirc),
        "disagreementByMode": {m: {"disagree": sum(1 for r in rows if r["mode"] == m and not r["agree"]),
                                    "total": sum(1 for r in rows if r["mode"] == m)} for m in MODES},
        "scorerStateOfDisagreements": dict(stc),
        "sampleDisagreements": samples,
        "honestBound": ("Per-answer fabricated? labels re-derived from the deterministic scorer and the Claude "
                        "judge (llmhub relay) on the SAME 108 abstain-pack answers. Characterizes WHERE they "
                        "disagree; does NOT validate either label as ground truth (no human gold standard)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
