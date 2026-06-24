#!/usr/bin/env python3
"""Validation pass for CPQA: bootstrap confidence intervals + control-flow sweep.

CPQA is scored by exact match (assert/abstain vs the pre-registered expectation), so
there is no judge subjectivity — the multi-judge / inter-rater step that gates *graded*
benchmarks does not apply to this deterministic core (it would apply to the future
LLM-controller path, where answers are generated). What a deterministic benchmark still
owes RESULTS.md-grade rigor is: (1) confidence intervals reflecting the finite query
sample, and (2) robustness across controller settings. This pass provides both.

- Bootstrap CI (B resamples of the query set) on each system's accuracy. For a system
  with zero observed errors we also report the rule-of-three upper bound (3/n).
- Control-flow sweep: the oracle→lexical accuracy gap across routing thresholds.

Stays candidateOnly / validated:false. Writes a validation report artifact.

    python tools/run_continual_qa_validation.py --episodes eval/continual_qa/episodes_v2_wiki.jsonl
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

from agent.continual_qa import control_flow_report, load_episodes, run_benchmark  # noqa: E402
from agent.continual_qa_controller import LexicalController  # noqa: E402
from agent.public_sanitize import sanitize_public_artifact  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "continual-qa.validation-report.json"


def bootstrap_ci(rows, key: str, *, B: int = 2000, seed: int = 12345) -> "dict":
    n = len(rows)
    rnd = random.Random(seed)
    accs: list[float] = []
    for _ in range(B):
        correct = sum(1 for _ in range(n) if rows[rnd.randrange(n)][key] == "correct")
        accs.append(correct / n)
    accs.sort()
    observed_errors = sum(1 for r in rows if r[key] != "correct")
    out = {
        "n": n,
        "pointAccuracy": round(sum(1 for r in rows if r[key] == "correct") / n, 4),
        "bootstrapMean": round(sum(accs) / B, 4),
        "ci95": [round(accs[int(0.025 * B)], 4), round(accs[min(int(0.975 * B), B - 1)], 4)],
        "observedErrors": observed_errors,
    }
    if observed_errors == 0:
        out["ruleOfThreeUpperErrorRate"] = round(3.0 / n, 4)   # 95% upper bound when 0/n seen
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)
    base = run_benchmark(episodes)
    rows = base["rows"]

    sweep = []
    for threshold in (1, 2, 3):
        cf = control_flow_report(episodes, LexicalController(min_overlap=threshold))
        sweep.append({"minOverlap": threshold, "endToEndAccuracy": cf["endToEndAccuracy"],
                      "controlFlowGap": cf["controlFlowGap"], "routingErrors": len(cf["routingErrors"])})

    report = {
        "schema": "sophia.continual_qa_validation.v1",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "note": ("Exact-match scoring => no judge subjectivity; LLM-judge families apply "
                 "to the future LLM-controller path, not this deterministic core."),
        "episodes": base["episodes"],
        "queryCount": len(rows),
        "accuracyCI": {
            "graph_backed": bootstrap_ci(rows, "graph_backed", B=args.bootstrap),
            "parametric_baseline": bootstrap_ci(rows, "parametric_baseline", B=args.bootstrap),
        },
        "retention": base["retention"],
        "controlFlowSweep": sweep,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "out": args.out,
        "graph_backed_CI": report["accuracyCI"]["graph_backed"]["ci95"],
        "graph_backed_ruleOfThree": report["accuracyCI"]["graph_backed"].get("ruleOfThreeUpperErrorRate"),
        "baseline_CI": report["accuracyCI"]["parametric_baseline"]["ci95"],
        "controlFlowSweep": sweep,
    }, indent=2))


if __name__ == "__main__":
    main()
