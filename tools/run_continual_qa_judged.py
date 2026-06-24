#!/usr/bin/env python3
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

from agent.continual_qa import GraphBackedSystem, load_episodes  # noqa: E402
from agent.continual_qa_answer import (  # noqa: E402
    build_source_map, cohen_kappa, generate_grounded, generate_raw, judge_answer, verdict,
)
from agent.deepseek_llm import make_complete  # noqa: E402
from agent.public_sanitize import sanitize_public_artifact  # noqa: E402
from okf.page import load_pages  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "continual-qa.judged.json"


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
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)
    source_map = build_source_map(load_pages(args.wiki))
    selected = _select(episodes, args.limit)

    answer_model = make_complete(model="deepseek-chat", max_tokens=256)
    judges = [
        ("deepseek-reasoner", make_complete(model="deepseek-reasoner", max_tokens=1500)),
        ("deepseek-chat", make_complete(model="deepseek-chat", max_tokens=96)),
    ]

    rows = []
    for q, expect, grounded in selected:
        src = source_map.get(q.target) if grounded else None
        answers = {
            "grounded": generate_grounded(q.text, src, answer_model),
            "raw": generate_raw(q.text, answer_model),
        }
        row = {"query": q.id, "target": q.target, "expect": expect, "type": q.type, "judges": {}}
        for system, ans in answers.items():
            row[f"answer_{system}"] = ans[:240]
            row["judges"][system] = {}
            for jname, jcomplete in judges:
                rating = judge_answer(q.text, ans, jcomplete)
                row["judges"][system][jname] = verdict(rating, expect)
        rows.append(row)
        print(f"{q.id} ({expect}): " + " ".join(
            f"{s}[{'/'.join('P' if row['judges'][s][jn] else 'F' for jn, _ in judges)}]"
            for s in ("grounded", "raw")))

    # Aggregate: per-judge and consensus (all judges agree pass) pass rates, + κ.
    summary = {}
    for system in ("grounded", "raw"):
        per_judge = {jn: round(sum(r["judges"][system][jn] for r in rows) / len(rows), 4)
                     for jn, _ in judges}
        consensus = [all(r["judges"][system][jn] for jn, _ in judges) for r in rows]
        ja = [r["judges"][system][judges[0][0]] for r in rows]
        jb = [r["judges"][system][judges[1][0]] for r in rows]
        summary[system] = {
            "perJudgePassRate": per_judge,
            "consensusPassRate": round(sum(consensus) / len(rows), 4),
            "consensusCI95": _bootstrap_ci(consensus),
            "interJudgeKappa": cohen_kappa(ja, jb),
        }

    report = {
        "schema": "sophia.continual_qa_judged.v1",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "answerModel": "deepseek-chat",
        "judges": [jn for jn, _ in judges],
        "distinctProviderFamilies": False,
        "caveats": [
            "Both judges are DeepSeek (same provider) -> validated:false; a RESULTS.md run needs >=2 provider families.",
            "deepseek-chat both generates answers and judges (self-grading risk).",
            "abstain-rubric: confident refutation of a fictional premise is scored as non-abstention, "
            "which can understate the raw model on fictional-premise controls; the grounded system's "
            "target behavior is uniform fail-closed abstention. The unambiguous contrast is on RETRACTED "
            "real facts, where raw asserts the stale fact and grounded abstains.",
        ],
        "queryCount": len(rows),
        "summary": summary,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
