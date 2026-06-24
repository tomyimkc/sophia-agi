#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judged-answer CPQA pass: grounded vs raw answers, multi-judge panel.

Generates a prose answer per question from two systems — ``grounded`` (answer only from
the retrieved OKF/wiki source) and ``raw`` (plain LLM, no source) — then scores both with
a panel of LLM judges and reports per-judge pass rates, the consensus (all-judges-agree)
pass rate with a bootstrap CI, and inter-judge Cohen's κ.

Grounding uses the GOLD target (oracle routing) so this isolates the *faithfulness* axis
from the *routing* axis (measured separately in run_continual_qa_llm.py).

    DEEPSEEK_API_KEY=... python tools/run_continual_qa_judged.py --limit 12

HONEST CAVEATS (kept in the report): both judges are the same provider (DeepSeek), so
``distinctProviderFamilies`` is false — a RESULTS.md-grade run needs ≥2 *provider*
families. One judge (deepseek-chat) also generates the answers (self-grading risk). This
is candidate machinery; swap in a second-provider judge to clear the gate. Network-only;
never in CI.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import deepseek_llm, llmhub_llm, openrouter_client  # noqa: E402
from agent.continual_qa import GraphBackedSystem, load_episodes  # noqa: E402
from agent.continual_qa_answer import (  # noqa: E402
    build_neighborhood_source_map, build_source_map, cohen_kappa, generate_grounded,
    generate_raw, judge_answer, percent_agreement, verdict,
)
from agent.continual_qa_hybrid import hybrid_answer  # noqa: E402
from tools.audit_cpqa_recall import classify_source  # noqa: E402
from agent.public_sanitize import sanitize_public_artifact  # noqa: E402
from okf.page import load_pages  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "continual-qa.judged.json"

# Coarse provider family per model id, for the distinct-families check.
def _family(model: str) -> str:
    m = model.lower()
    canon = {"claude": "anthropic", "gemini": "google", "gpt": "openai",
             "llama": "meta", "mistral": "mistral"}
    for fam in ("claude", "gemini", "gpt", "deepseek", "llama", "mistral", "qwen", "grok", "doubao", "kimi"):
        if fam in m:
            return canon.get(fam, fam)
    return model


def _complete_for(spec: str, *, max_tokens: int):
    """spec = 'provider:model'. provider in {llmhub, deepseek, openrouter}.

    Model ids may contain ':' (e.g. openrouter:meta-llama/llama-3.3-70b-instruct:free),
    so split only on the first ':'."""
    providers = {"llmhub": llmhub_llm, "deepseek": deepseek_llm, "openrouter": openrouter_client}
    provider, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"invalid model spec {spec!r}: expected 'provider:model' (e.g. 'deepseek:deepseek-chat')")
    if provider not in providers:
        raise ValueError(f"unknown provider {provider!r} in spec {spec!r}: choose one of {sorted(providers)}")
    module = providers[provider]
    return model, module.make_complete(model=model, max_tokens=max_tokens)


def _select(episodes, limit):
    """Replay the stream; return (query, expect, grounded_bool) keeping all abstain
    queries plus enough assert queries to reach ``limit`` (a mix of both)."""
    gb = GraphBackedSystem()
    items = []
    for ep in episodes:
        gb.learn(ep.learn)
        gb.retract(ep.retract)
        state = gb.grounded_state()
        for q in ep.queries:
            items.append((q, q.expect, q.target in state))
    abstains = [it for it in items if it[1] == "abstain"]
    asserts = [it for it in items if it[1] == "assert"]
    keep = abstains + asserts[: max(0, limit - len(abstains))]
    return keep[:limit]


def _bootstrap_ci(flags, B=2000, seed=7):
    n = len(flags)
    if n == 0:
        return [0.0, 0.0]
    rnd = random.Random(seed)
    rates = sorted(sum(flags[rnd.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [round(rates[int(0.025 * B)], 4), round(rates[min(int(0.975 * B), B - 1)], 4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--wiki", default=str(ROOT / "wiki"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--answer", default="llmhub:gpt-5-mini", help="provider:model for answers")
    ap.add_argument("--grounded-mode", default="strict", choices=["strict", "attribution_safe"],
                    help="Step 4: 'attribution_safe' allows general-fact recall while keeping attribution discipline")
    ap.add_argument("--retrieval", default="single", choices=["single", "neighborhood"],
                    help="Step 1: 'neighborhood' backs each page with its k-hop OKF neighbors")
    ap.add_argument("--hops", type=int, default=1, help="neighborhood hop count (Step 1)")
    ap.add_argument("--grounded-system", default="grounded", choices=["grounded", "hybrid"],
                    help="Step 2: 'hybrid' routes by context type (abstain / strict / attribution-safe fallback)")
    ap.add_argument("--judge", action="append", default=None,
                    help="provider:model judge (repeatable); default = cross-family Claude + Gemini")
    args = ap.parse_args()

    judge_specs = args.judge or ["llmhub:claude-opus-4-8", "llmhub:gemini-2.5-pro"]
    episodes = load_episodes(args.episodes)
    pages = load_pages(args.wiki)
    source_map = (build_neighborhood_source_map(pages, hops=args.hops)
                  if args.retrieval == "neighborhood" else build_source_map(pages))
    selected = _select(episodes, args.limit)

    answer_model_id, answer_complete = _complete_for(args.answer, max_tokens=256)
    judges = [_complete_for(spec, max_tokens=800) for spec in judge_specs]   # (model, complete)
    judge_models = [m for m, _ in judges]
    judge_families = sorted({_family(m) for m in judge_models})
    answer_family = _family(answer_model_id)

    # Step 2: per-target source sufficiency (answer-bearing prose vs thin stub).
    answer_bearing = {p.id: classify_source(p)["answerBearing"] for p in pages}
    policy_counts: dict[str, int] = {}

    def grounded_answer(q, src):
        if args.grounded_system == "hybrid":
            ans, policy = hybrid_answer(q.text, src, answer_complete,
                                        answer_bearing=answer_bearing.get(q.target, False))
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
            return ans
        return generate_grounded(q.text, src, answer_complete, mode=args.grounded_mode)

    run_results = []
    for r in range(args.runs):
        rows = []
        for q, expect, grounded in selected:
            src = source_map.get(q.target) if grounded else None
            answers = {"grounded": grounded_answer(q, src),
                       "raw": generate_raw(q.text, answer_complete)}
            row = {"query": q.id, "expect": expect, "judges": {}}
            for system, ans in answers.items():
                row["judges"][system] = {m: verdict(judge_answer(q.text, ans, jc), expect)
                                         for m, jc in judges}
            rows.append(row)
        run_results.append(rows)
        print(f"run {r + 1}/{args.runs} done ({len(rows)} queries)")

    def agg(system: str) -> "dict":
        per_run = []
        pooled_consensus = []
        by_expect: dict = {"assert": [], "abstain": []}
        kappas, agreements = [], []
        for rows in run_results:
            consensus = [all(row["judges"][system][m] for m in judge_models) for row in rows]
            pooled_consensus.extend(consensus)
            for row, c in zip(rows, consensus):
                by_expect.setdefault(row["expect"], []).append(c)
            per_judge = {m: round(sum(row["judges"][system][m] for row in rows) / len(rows), 4)
                         for m in judge_models}
            entry = {"consensusPassRate": round(sum(consensus) / len(rows), 4),
                     "perJudgePassRate": per_judge, "interJudgeKappa": None,
                     "interJudgePercentAgreement": None}
            # Mean pairwise inter-judge agreement over ALL judge pairs — generalizes the
            # 2-judge case to an N-family panel (validation across >2 families).
            if len(judge_models) >= 2:
                votes = {m: [row["judges"][system][m] for row in rows] for m in judge_models}
                pair_k, pair_a = [], []
                for i in range(len(judge_models)):
                    for j in range(i + 1, len(judge_models)):
                        pair_k.append(cohen_kappa(votes[judge_models[i]], votes[judge_models[j]]))
                        pair_a.append(percent_agreement(votes[judge_models[i]], votes[judge_models[j]]))
                kappas.append(round(sum(pair_k) / len(pair_k), 4))
                agreements.append(round(sum(pair_a) / len(pair_a), 4))
                entry["interJudgeKappa"] = kappas[-1]
                entry["interJudgePercentAgreement"] = agreements[-1]
            per_run.append(entry)
        return {
            "perRun": per_run,
            "meanConsensusPassRate": round(sum(pooled_consensus) / len(pooled_consensus), 4),
            "consensusCI95": _bootstrap_ci(pooled_consensus),
            # Where grounding actually helps: split consensus pass by expectation. The
            # recall ("assert") cases are facts a strong raw model also knows; the
            # fail-closed advantage of grounding concentrates on the "abstain" traps.
            "byExpect": {k: {"n": len(v), "consensusPassRate": round(sum(v) / len(v), 4)}
                         for k, v in by_expect.items() if v},
            "meanInterJudgeKappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
            "meanInterJudgePercentAgreement": round(sum(agreements) / len(agreements), 4) if agreements else None,
        }

    summary = {"grounded": agg("grounded"), "raw": agg("raw")}
    report = {
        "schema": "sophia.continual_qa_judged.v2",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "answerModel": answer_model_id,
        "answerFamily": answer_family,
        "judges": judge_models,
        "judgeFamilies": judge_families,
        "distinctProviderFamilies": len(judge_families) >= 2,
        "selfGradingRisk": answer_family in judge_families,
        "groundedMode": args.grounded_mode,
        "groundedSystem": args.grounded_system,
        "retrieval": args.retrieval,
        "hops": args.hops if args.retrieval == "neighborhood" else None,
        "hybridPolicyCounts": policy_counts or None,
        "runs": args.runs,
        "queryCount": len(selected),
        "summary": summary,
        "caveats": [
            "Models reached via one gateway (LLMHub) under one key; validated:false pending "
            "independent replication and pre-registration, though judges are distinct families.",
            "abstain-rubric scores confident refutation of a fictional premise as non-abstention, "
            "which can understate the raw model on fictional-premise controls; the unambiguous "
            "contrast is on RETRACTED real facts (raw asserts the stale fact, grounded abstains).",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({"judges": judge_models, "judgeFamilies": judge_families,
                      "distinctProviderFamilies": report["distinctProviderFamilies"],
                      "selfGradingRisk": report["selfGradingRisk"], "summary": summary}, indent=2))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
