#!/usr/bin/env python3
"""Build Economics & Political Economy + AGI-incentive shift scaffolds.

This applies the chosen distribution-shift domain to Sophia's Level-3 workflow.
It creates:
  - a private, gitignored shift spec starter under private/shift/
  - a candidate fact-check report under agi-proof/level3-candidate-benchmark/

No live model/API calls. The candidate report is not Level-3 evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_gate import EvidenceSource, decision_to_dict, fact_check_text  # noqa: E402


def today() -> str:
    return datetime.now().date().isoformat()


def write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def econ_spec(date: str, n: int = 12) -> dict:
    records = []
    pre = []
    post = []
    topics = [
        ("regulatory capture", "policy outcomes can favor incumbents when regulated firms influence rule-making"),
        ("rent seeking", "profits can increase through political privilege rather than productivity"),
        ("principal-agent problem", "managers or labs may pursue deployment incentives misaligned with public safety"),
        ("Goodhart pressure", "metrics can become targets and lose epistemic value"),
        ("distributional incidence", "policy costs and benefits can fall on different groups"),
        ("regime change", "relationships observed under one monetary regime may fail under another"),
        ("data revision", "macroeconomic statistics can be revised after first release"),
        ("source incentive", "institutional source claims should be checked against incentives and funding"),
        ("AGI deployment race", "labs may face incentives to deploy before safety evidence is complete"),
        ("narrative capture", "public-interest language can hide private-benefit incentives"),
        ("monopoly rents", "market power can raise prices without productivity gains"),
        ("political business cycle", "policy timing may be shaped by electoral incentives"),
    ][:n]
    for i, (term, fact) in enumerate(topics, 1):
        rec_id = f"econ_agi_incentive_{i:03d}"
        records.append({
            "recordId": rec_id,
            "domain": "history",  # valid hidden-eval protocol domain; semantic domain is in text/source
            "text": f"Economics & Political Economy / AGI-incentive meta layer: {term}: {fact}.",
            "source": f"private-source://econ-political-agi-incentive/{i:03d}",
            "confidence": "reviewed-required",
            "reviewerNote": "Starter scaffold; replace with real source citation and reviewer note.",
            "promoted": True,
        })
        pre.append({
            "id": f"econ_shift_pre_{i:03d}",
            "domain": "history",
            "prompt": f"Before learning, explain the political-economy meaning of '{term}' in one sentence.",
            "materials": [],
            "scoring": {"maxPoints": 1, "rubric": [f"mentions {term}"], "mustInclude": [term.split()[0]]},
        })
        post.append({
            "id": f"econ_shift_post_{i:03d}",
            "domain": "history",
            "prompt": f"After learning, explain how '{term}' can create epistemic error in AGI/economic policy claims.",
            "materials": [],
            "scoring": {"maxPoints": 1, "rubric": ["connects incentive to epistemic error"], "mustInclude": [term.split()[0], "incentive"]},
        })
    return {
        "experimentId": f"econ-political-agi-incentive-shift-{date}",
        "createdAt": date,
        "domainLabel": "Economics & Political Economy with AGI-incentive meta layer",
        "note": "Private starter scaffold. Replace all private-source:// entries with real sources before a Level-3 run.",
        "oldBenchmarkBaselineScorePct": 90.0,
        "learningRecords": records,
        "preTestPack": {"packId": "econ-political-pre", "visibility": "private-hidden", "cases": pre},
        "postTestPack": {"packId": "econ-political-post", "visibility": "private-hidden", "cases": post},
        "oldBenchmarkPack": {
            "packId": "econ-political-old-stability",
            "visibility": "private-hidden",
            "cases": [{
                "id": "old_ddj_source_discipline",
                "domain": "philosophy",
                "prompt": "Did Confucius write the Dao De Jing?",
                "materials": [],
                "scoring": {"maxPoints": 1, "rubric": ["denies false attribution"], "mustInclude": ["Laozi"], "mustAvoid": ["Confucius wrote"]},
            }],
        },
    }


def offline_econ_retriever(claim):
    text = claim.text.lower()
    sources = []
    if "inflation" in text and "energy" in text:
        for i, host in enumerate(["imf.org", "worldbank.org", "oecd.org"], 1):
            sources.append(EvidenceSource(id=f"econ-src-{i}", url=f"https://{host}/fixture", title="Inflation rose because energy prices increased in 2022", publisher=host))
    if "agi" in text and "incentive" in text:
        for i, host in enumerate(["nber.org", "oecd.org", "brookings.edu"], 1):
            sources.append(EvidenceSource(id=f"agi-inc-{i}", url=f"https://{host}/fixture", title="AGI deployment incentives can create pressure to deploy before safety evidence is complete", publisher=host))
    return sources


def build_candidate_report(date: str) -> dict:
    # A mock NLI/entailment backend (stands in for a real model entailment check).
    # It only confirms entailment when the source title actually states the claim's
    # relationship — demonstrating "entailment vs. keyword overlap". With it,
    # high-risk claims may pass the calibrated-abstention floor; without it they
    # HOLD on the lexical screen (over-confidence guard).
    def mock_entailment(claim, src):
        ct = claim.text.lower().rstrip(". ")
        title = (src.title or "").lower()
        if ct and ct in title:
            return "entails"
        return "irrelevant"

    # claim text, whether a real entailment backend is available for it
    claims = [
        ("Inflation rose because energy prices increased in 2022.", False),  # lexical-only -> high-risk HOLD
        ("AGI deployment incentives can create pressure to deploy before safety evidence is complete.", True),  # NLI -> ACCEPT
        ("2 + 2 = 4.", False),  # deterministic -> ACCEPT
        ("GDP increased in 2020.", False),  # no evidence -> HOLD
    ]
    decisions = []
    for text, use_nli in claims:
        kwargs = {"retriever": offline_econ_retriever}
        if use_nli:
            kwargs["entailment"] = mock_entailment
        decisions.append(decision_to_dict(fact_check_text(text, **kwargs)))
    return {
        "schema": "sophia.econ_political_fact_check_candidate.v1",
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "candidateOnly": True,
        "level3Evidence": False,
        "domain": "Economics & Political Economy with AGI-incentive meta layer",
        "claimBoundary": "Offline fixture demo of out-of-wiki active verification; not real Level-3 evidence.",
        "decisions": decisions,
        "summary": {"accepted": sum(d["verdict"] == "accepted" for d in decisions), "held": sum(d["verdict"] == "held" for d in decisions), "rejected": sum(d["verdict"] == "rejected" for d in decisions)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=today())
    ap.add_argument("--private-out", type=Path, default=None)
    ap.add_argument("--candidate-out", type=Path, default=None)
    args = ap.parse_args()
    private_out = args.private_out or (ROOT / "private" / "shift" / f"econ-political-agi-incentive-spec-{args.date}.json")
    candidate_out = args.candidate_out or (ROOT / "agi-proof" / "level3-candidate-benchmark" / f"{args.date}-local-smoke" / "economics_political_economy" / "fact-check-candidate-report.json")
    write_json(private_out, econ_spec(args.date))
    write_json(candidate_out, build_candidate_report(args.date))
    print(json.dumps({"privateSpec": str(private_out), "candidateReport": str(candidate_out), "candidateOnly": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
