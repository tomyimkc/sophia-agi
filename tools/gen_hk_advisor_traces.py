#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate verified HK bilingual advisor SFT traces (Phase 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hk_advisor.policy import HK_ADVISOR_SYSTEM, format_response  # noqa: E402
from agent.hk_advisor.verifier import trace_passes, verify_trace  # noqa: E402
from provenance_bench.dataset_guard import hk_advisor_benchmark_prompt_set, normalize  # noqa: E402

OUT_DEFAULT = ROOT / "training" / "hk_advisor" / "sft_traces.jsonl"

# Hand-crafted verified responses keyed by council task id (disjoint from held-out benchmark).
_ANSWERS: dict[str, tuple[str, str, list[str]]] = {
    "law_pdpo_transfer": (
        "en",
        "answerable",
        "Section 33 of the Personal Data (Privacy) Ordinance (Cap. 486) restricts transfers "
        "of personal data outside Hong Kong, but s.33 is **not yet in force**. Other PDPO "
        "principles still apply to cross-border transfers in practice.",
        ["Cap. 486", "PDPO", "section 33"],
    ),
    "law_employment_termination": (
        "en",
        "answerable",
        "Under the Employment Ordinance, an employer terminating a continuous contract "
        "must give at least 7 days' notice or payment in lieu.",
        ["7 days", "continuous contract"],
    ),
    "law_consumer_refund": (
        "en",
        "answerable",
        "A blanket 'no refund' notice cannot exclude implied quality conditions under the "
        "Sale of Goods Ordinance for defective goods sold to consumers.",
        ["Sale of Goods", "defective"],
    ),
    "law_director_duty": (
        "en",
        "answerable",
        "Directors owe fiduciary duties and the duty to exercise reasonable care, skill and "
        "diligence under the Companies Ordinance.",
        ["Companies Ordinance", "director"],
    ),
    "law_data_breach_notify": (
        "en",
        "abstain",
        "There is no standalone mandatory PCPD notification deadline in force identical to GDPR; "
        "notification practice depends on sector guidance and facts. I cannot confirm a fixed "
        "legal duty without verified primary sources for your sector.",
        [],
    ),
    "fin_runway": (
        "en",
        "answerable",
        "Runway = HK$1,000,000 / HK$50,000 per month = 20 months.",
        ["20 months"],
    ),
    "fin_gross_margin": (
        "en",
        "answerable",
        "Gross margin = (HK$200,000 - HK$140,000) / HK$200,000 = 30%.",
        ["30%"],
    ),
    "fin_rule40": (
        "en",
        "answerable",
        "Rule of 40 score = 30% growth + (-15%) margin = 15%, which does NOT meet the 40% threshold.",
        ["Rule of 40", "15%"],
    ),
    "fin_cac_ltv": (
        "en",
        "answerable",
        "LTV:CAC = HK$900 / HK$300 = 3:1, generally considered healthy for SaaS.",
        ["3:1"],
    ),
    "fin_pricing_change": (
        "en",
        "answerable",
        "Net revenue effect = 1.20 × 0.90 = 1.08, i.e. +8% revenue.",
        ["8%"],
    ),
    "econ_minwage": (
        "en",
        "answerable",
        "A 10% minimum-wage rise on HK$40/hour → HK$44/hour (+HK$4/hour for a full-time worker).",
        ["HK$44", "10%"],
    ),
    "econ_gig_classification": (
        "en",
        "abstain",
        "Employee vs contractor classification in Hong Kong depends on control, integration, and "
        "economic reality; I cannot confirm your platform's classification without case-specific "
        "facts and verified authority.",
        [],
    ),
    "law_lease_forfeit": (
        "en",
        "abstain",
        "Commercial lease forfeiture in Hong Kong typically requires compliance with the lease and "
        "often court relief under established principles; I cannot confirm whether forfeiture without "
        "a court order is available on your facts without verified case-specific authority.",
        [],
    ),
    "fin_burn_multiple": (
        "en",
        "answerable",
        "Burn multiple = net burn / net new ARR = HK$400,000 / HK$200,000 = 2.0× (efficient if ≤1.5×).",
        ["2.0", "burn multiple"],
    ),
    "fin_dilution": (
        "en",
        "answerable",
        "Rough dilution ≈ investment / (cap + investment) = HK$2M / (HK$20M + HK$2M) ≈ 9.1% pre-money SAFE.",
        ["9.1%", "dilution"],
    ),
    "econ_tariff": (
        "en",
        "answerable",
        "A 15% import tariff is partly borne by importers and partly passed to consumers depending on "
        "elasticity; options include absorb margin, re-source, or raise prices.",
        ["15%", "tariff"],
    ),
    "econ_unit_economics": (
        "en",
        "answerable",
        "Fast growth with negative unit economics (LTV:CAC < 1 or gross margin too low) is economically "
        "unhealthy — watch burn multiple and payback period.",
        ["unit economics", "burn"],
    ),
}

_SYNTHETIC = [
    {
        "id": "syn-yue-pdpo-001",
        "prompt": "香港 PDPO Cap. 486 第33條而家生效未？跨境傳送個人資料要注意咩？",
        "language": "yue",
        "label": {"decision": "answerable", "gold_citations": ["Cap. 486"], "mustInclude": ["第33條", "未生效"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "body": "第33條限制將個人資料傳送至香港以外，但而家仍未生效（Cap. 486 PDPO）。其他資料保護原則仍然適用。",
    },
    {
        "id": "syn-yue-runway-001",
        "prompt": "現金100萬，每月burn 5萬，runway 有幾耐？",
        "language": "yue",
        "label": {"decision": "answerable", "gold_citations": [], "mustInclude": ["20"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "body": "Runway = 1,000,000 / 50,000 = 20個月。",
    },
    {
        "id": "syn-en-abstain-future-001",
        "prompt": "Will Hong Kong pass an AI liability bill in 2028? Give the exact Cap. number.",
        "language": "en",
        "label": {"decision": "abstain", "gold_citations": [], "mustInclude": []},
        "trap": "none",
        "epistemicLabel": "abstain",
        "body": "I cannot confirm whether Hong Kong will enact an AI liability bill in 2028 or assign "
                "a Cap. number — no verified primary source exists for unreleased legislation.",
    },
    {
        "id": "syn-yue-abstain-future-001",
        "prompt": "2028年香港會唔會有 AI 責任條例？條例編號係咩？",
        "language": "yue",
        "label": {"decision": "abstain", "gold_citations": [], "mustInclude": []},
        "trap": "none",
        "epistemicLabel": "abstain",
        "body": "我無法確認2028年會否有 AI 責任條例或條例編號 — 未公布嘅立法冇可靠來源。",
    },
    {
        "id": "syn-en-life-deposit-001",
        "prompt": "What is the typical residential tenancy deposit in Hong Kong?",
        "language": "en",
        "label": {"decision": "answerable", "gold_citations": [], "mustInclude": ["two months"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "body": "Two months' rent is the common market practice for residential deposits in Hong Kong.",
    },
    {
        "id": "syn-en-cap614-001",
        "prompt": "Does Cap. 614 give e-Legislation copies legal status in Hong Kong?",
        "language": "en",
        "label": {"decision": "answerable", "gold_citations": ["Cap. 614"], "mustInclude": ["Cap. 614"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "body": "Yes — the Legislation Publication Ordinance (Cap. 614) gives verified e-Legislation copies legal status.",
    },
    {
        "id": "syn-en-hkcfi-001",
        "prompt": "What did [2025] HKCFI 808 warn about AI-drafted submissions?",
        "language": "en",
        "label": {"decision": "answerable", "gold_citations": ["[2025] HKCFI 808"], "mustInclude": ["HKCFI 808"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "body": "The court criticised AI-drafted submissions for making up allegations without evidential basis ([2025] HKCFI 808).",
    },
    {
        "id": "syn-yue-cap614-001",
        "prompt": "Cap. 614 係咪話電子版法例有法律效力？",
        "language": "yue",
        "label": {"decision": "answerable", "gold_citations": ["Cap. 614"], "mustInclude": ["Cap. 614"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "body": "係，《法例發布條例》（Cap. 614）賦予經核實嘅 e-Legislation 電子文本法律效力。",
    },
    {
        "id": "syn-yue-margin-001",
        "prompt": "收入20萬，COGS 14萬，gross margin 係幾多？",
        "language": "yue",
        "label": {"decision": "answerable", "gold_citations": [], "mustInclude": ["30%"]},
        "trap": "none",
        "epistemicLabel": "definite",
        "body": "Gross margin = (200,000 - 140,000) / 200,000 = 30%。",
    },
    {
        "id": "syn-en-abstain-order-001",
        "prompt": "What exact refund will Shop XYZ offer for order #99999?",
        "language": "en",
        "label": {"decision": "abstain", "gold_citations": [], "mustInclude": []},
        "trap": "none",
        "epistemicLabel": "abstain",
        "body": "I cannot confirm Shop XYZ's refund policy for an unknown order — no verified source.",
    },
]


def _load_council_tasks() -> list[dict]:
    path = ROOT / "data" / "council_tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("tasks", []))


def _task_pool() -> list[dict]:
    forbidden = hk_advisor_benchmark_prompt_set(root=ROOT)
    pool: list[dict] = []
    for task in _load_council_tasks():
        prompt = task["prompt"]
        if normalize(prompt) in forbidden:
            continue
        tid = task["id"]
        if tid in _ANSWERS:
            lang, decision, body, must = _ANSWERS[tid]
            pool.append({
                "id": tid,
                "prompt": prompt,
                "language": lang,
                "label": {"decision": decision, "gold_citations": [], "mustInclude": must},
                "trap": "none",
                "epistemicLabel": "definite" if decision == "answerable" else "abstain",
                "scoring": {"mustInclude": must},
                "body": body,
            })
    for syn in _SYNTHETIC:
        if normalize(syn["prompt"]) not in forbidden:
            pool.append(syn)
    return pool


def _to_messages(prompt: str, answer: str) -> list[dict]:
    return [
        {"role": "system", "content": HK_ADVISOR_SYSTEM},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def generate_traces(*, mode: str, max_traces: int, out: Path) -> dict:
    pool = _task_pool()
    kept, dropped = [], {"verify_fail": 0, "decontam": 0}
    for case in pool:
        if len(kept) >= max_traces:
            break
        answer = format_response(case["body"], language=case.get("language", "en"))
        verdicts = verify_trace(answer=answer, case=case)
        if not trace_passes(verdicts):
            dropped["verify_fail"] += 1
            continue
        kept.append({
            "messages": _to_messages(case["prompt"], answer),
            "metadata": {
                "caseId": case["id"],
                "language": case.get("language", "en"),
                "decision": case["label"]["decision"],
                "verified": True,
                "mode": mode,
                "candidateOnly": True,
                "canClaimAGI": False,
            },
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"kept": len(kept), "dropped": dropped, "poolSize": len(pool), "out": str(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--max-traces", type=int, default=80)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    stats = generate_traces(mode=args.mode, max_traces=args.max_traces, out=args.out)
    print(json.dumps(stats, indent=2))
    return 0 if stats["kept"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
