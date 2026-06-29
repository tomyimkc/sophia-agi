#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mine abstention-calibration DPO pairs for HK advisor (Phase 3)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hk_advisor.policy import format_response  # noqa: E402
from provenance_bench.dataset_guard import hk_advisor_benchmark_prompt_set, normalize  # noqa: E402
from provenance_bench.hk_advisor_benchmark import load_cases  # noqa: E402

OUT_DEFAULT = ROOT / "training" / "hk_advisor" / "dpo_pairs.jsonl"

RANK = {
    "correct_cited_answer": 0,
    "correct_abstain": 1,
    "wrong_abstain": 2,
    "fabricated_regulation": 4,
    "fake_citation": 4,
    "overconfident_trap": 4,
    "uncited_claim": 4,
}


def _good_answerable(case: dict) -> str:
    gold = case.get("label", {}).get("gold_answer") or ""
    cites = case.get("label", {}).get("gold_citations") or []
    cite_str = f" ({', '.join(cites)})" if cites else ""
    body = f"{gold}{cite_str}".strip() or "See verified sources above."
    return format_response(body, language=case.get("language", "en"))


def _good_abstain(case: dict) -> str:
    lang = case.get("language", "en")
    if lang == "yue":
        body = "我無法確認呢個問題 — 冇可靠來源或超出可核實範圍。"
    else:
        body = "I cannot confirm this from verified sources; the answer remains unknown or disputed."
    return format_response(body, language=lang)


def _rejected_specs(case: dict) -> list[tuple[str, str]]:
    lang = case.get("language", "en")
    decision = case.get("label", {}).get("decision", "")
    trap = case.get("trap", "none")
    markers = case.get("fabricationMarkers") or []
    specs: list[tuple[str, str]] = []

    if decision == "answerable":
        specs.append(("wrong_abstain", _good_abstain(case)))
        specs.append(("uncited_claim", format_response(
            "The answer is yes based on general principles.", language=lang)))
    elif trap == "fabrication_bait" and markers:
        fab = markers[0]
        specs.append(("fabricated_regulation", format_response(
            f"Under {fab}, the requirement is mandatory disclosure.", language=lang)))
        specs.append(("overconfident_trap", format_response(
            f"Yes — {fab} clearly applies to your situation.", language=lang)))
    elif trap == "fake_citation":
        fake = markers[0] if markers else "[2025] HKCFI 99999"
        specs.append(("fake_citation", format_response(
            f"Per {fake}, the court held that your position is correct.", language=lang)))
        specs.append(("overconfident_trap", format_response(
            f"The answer is established in {fake}.", language=lang)))
    elif trap == "unanswerable":
        specs.append(("overconfident_trap", format_response(
            "The answer is definitely X based on my knowledge.", language=lang)))
    else:
        specs.append(("overconfident_trap", format_response(
            "I can confirm the exact outcome with certainty.", language=lang)))
    return specs


def mine_pairs(*, out: Path, max_pairs: int = 400) -> dict:
    forbidden = hk_advisor_benchmark_prompt_set(root=ROOT)
    from tools.gen_hk_advisor_traces import _task_pool  # noqa: E402
    train_pool = [c for c in _task_pool() if normalize(c["prompt"]) not in forbidden]
    bench = load_cases()
    rng = random.Random(42)
    rng.shuffle(train_pool)

    pairs: list[dict] = []
    counts: dict[str, int] = {}

    for case in train_pool:
        if len(pairs) >= max_pairs:
            break
        decision = case.get("label", {}).get("decision", "")
        chosen = _good_answerable(case) if decision == "answerable" else _good_abstain(case)
        rank_chosen = RANK["correct_cited_answer"] if decision == "answerable" else RANK["correct_abstain"]
        for rtype, rejected in _rejected_specs(case):
            if len(pairs) >= max_pairs:
                break
            pairs.append({
                "prompt": case["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "rejected_type": rtype,
                    "caseId": case.get("id", ""),
                    "rank_chosen": rank_chosen,
                    "rank_rejected": RANK.get(rtype, 4),
                    "candidateOnly": True,
                    "canClaimAGI": False,
                },
            })
            counts[rtype] = counts.get(rtype, 0) + 1

    stub_templates = {
        "fabrication_bait": {
            "prompt": "[TRAINING-STUB] What does the unreleased Cap. 712 Section 42 require for AI liability in HK?",
            "fabricationMarkers": ["Cap. 712", "Section 42"],
        },
        "fake_citation": {
            "prompt": "[TRAINING-STUB] Summarise the holding in [2024] HKCFI 99999 on tenancy deposits.",
            "fabricationMarkers": ["[2024] HKCFI 99999"],
        },
        "unanswerable": {
            "prompt": "[TRAINING-STUB] What will the exact text of Hong Kong's unreleased 2028 AI governance code be?",
            "fabricationMarkers": ["2028", "AI governance code"],
        },
    }
    for trap_type, stub in stub_templates.items():
        case = {
            "language": "en", "trap": trap_type,
            "fabricationMarkers": stub["fabricationMarkers"],
            "label": {"decision": "abstain"},
            "epistemicLabel": "abstain",
        }
        chosen = _good_abstain(case)
        for rtype, rejected in _rejected_specs(case):
            if len(pairs) >= max_pairs:
                break
            pairs.append({
                "prompt": stub["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "rejected_type": rtype,
                    "caseId": f"stub-{trap_type}-{rtype}",
                    "rank_chosen": RANK["correct_abstain"],
                    "rank_rejected": RANK.get(rtype, 4),
                    "trapType": trap_type,
                    "candidateOnly": True,
                    "canClaimAGI": False,
                },
            })
            counts[rtype] = counts.get(rtype, 0) + 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    return {"pairs": len(pairs), "byRejectedType": counts, "trainPool": len(train_pool)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--max-pairs", type=int, default=400)
    args = ap.parse_args(argv)
    stats = mine_pairs(out=args.out, max_pairs=args.max_pairs)
    print(json.dumps(stats, indent=2))
    return 0 if stats["pairs"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
