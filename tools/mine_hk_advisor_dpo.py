#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mine abstention-calibration DPO pairs for HK advisor (Phase 3)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hk_advisor.policy import format_response  # noqa: E402
from provenance_bench.dataset_guard import hk_advisor_benchmark_prompt_set, normalize  # noqa: E402

OUT_DEFAULT = ROOT / "training" / "hk_advisor" / "dpo_pairs.jsonl"
RANK = {"correct_cited_answer": 0, "correct_abstain": 1, "wrong_abstain": 2,
        "fabricated_regulation": 4, "fake_citation": 4, "overconfident_trap": 4, "uncited_claim": 4}


def _good_answerable(case: dict) -> str:
    gold = case.get("label", {}).get("gold_answer") or ""
    cites = case.get("label", {}).get("gold_citations") or []
    s = f"{gold}{(' (' + cites[0] + ')') if cites else ''}".strip()
    return format_response(s or "See verified sources.", language=case.get("language", "en"))


def _good_abstain(case: dict) -> str:
    lang = case.get("language", "en")
    body = "我無法確認 — 冇可靠來源。" if lang == "yue" else "I cannot confirm; no verified source."
    return format_response(body, language=lang)


def _rejected_specs(case: dict) -> list[tuple[str, str]]:
    lang, decision = case.get("language", "en"), case.get("label", {}).get("decision", "")
    trap, markers = case.get("trap", "none"), case.get("fabricationMarkers") or []
    if decision == "answerable":
        return [("wrong_abstain", _good_abstain(case)),
                ("uncited_claim", format_response("Yes, based on general principles.", language=lang))]
    if trap == "fabrication_bait" and markers:
        f = markers[0]
        return [("fabricated_regulation", format_response(f"Under {f}, disclosure is mandatory.", language=lang)),
                ("overconfident_trap", format_response(f"Yes — {f} applies.", language=lang))]
    if trap == "fake_citation":
        f = markers[0] if markers else "[2025] HKCFI 99999"
        return [("fake_citation", format_response(f"Per {f}, you are correct.", language=lang)),
                ("overconfident_trap", format_response(f"Established in {f}.", language=lang))]
    return [("overconfident_trap", format_response("The answer is definitely X.", language=lang))]


def mine_pairs(*, out: Path, max_pairs: int = 400) -> dict:
    from tools.gen_hk_advisor_traces import _task_pool  # noqa: E402
    forbidden = hk_advisor_benchmark_prompt_set(root=ROOT)
    pairs, counts = [], {}
    for case in [c for c in _task_pool() if normalize(c["prompt"]) not in forbidden]:
        if len(pairs) >= max_pairs:
            break
        d = case.get("label", {}).get("decision", "")
        chosen = _good_answerable(case) if d == "answerable" else _good_abstain(case)
        rc = RANK["correct_cited_answer"] if d == "answerable" else RANK["correct_abstain"]
        for rt, rej in _rejected_specs(case):
            if len(pairs) >= max_pairs:
                break
            pairs.append({"prompt": case["prompt"], "chosen": chosen, "rejected": rej,
                          "metadata": {"rejected_type": rt, "caseId": case.get("id", ""),
                                       "rank_chosen": rc, "rank_rejected": RANK[rt],
                                       "candidateOnly": True, "canClaimAGI": False}})
            counts[rt] = counts.get(rt, 0) + 1
    for tt, stub in {"fabrication_bait": "[TRAINING-STUB] Cap. 712 Section 42 AI liability?",
                     "fake_citation": "[TRAINING-STUB] [2024] HKCFI 99999 holding?",
                     "unanswerable": "[TRAINING-STUB] Unreleased 2028 AI governance code text?"}.items():
        case = {"language": "en", "trap": tt,
                "fabricationMarkers": ["Cap. 712"] if tt == "fabrication_bait" else ["[2024] HKCFI 99999"],
                "label": {"decision": "abstain"}}
        for rt, rej in _rejected_specs(case):
            if len(pairs) >= max_pairs:
                break
            pairs.append({"prompt": stub, "chosen": _good_abstain(case), "rejected": rej,
                          "metadata": {"rejected_type": rt, "trapType": tt,
                                       "candidateOnly": True, "canClaimAGI": False}})
            counts[rt] = counts.get(rt, 0) + 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return {"pairs": len(pairs), "byRejectedType": counts}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--max-pairs", type=int, default=400)
    args = ap.parse_args(argv)
    stats = mine_pairs(out=args.out, max_pairs=args.max_pairs)
    print(json.dumps(stats, indent=2))
    return 0 if stats["pairs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
