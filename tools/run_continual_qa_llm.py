#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Live LLM-controller pass for CPQA — the real control-flow error.

Routes every CPQA question with a real LLM (DeepSeek, OpenAI-compatible) instead of the
oracle/lexical stand-ins, across N runs, and reports the true control-flow gap: how much
end-to-end accuracy the LLM-as-control-flow layer costs relative to the perfect-routing
substrate. This is the measurement limitation #1 demands but a deterministic harness
cannot make.

    DEEPSEEK_API_KEY=... python tools/run_continual_qa_llm.py --runs 3

Requires network + key (see agent/deepseek_llm.py). Never run in CI. The written report
contains only metrics — no key, no raw model text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa import control_flow_report, load_episodes  # noqa: E402
from agent.continual_qa_controller import LLMController  # noqa: E402
from agent.deepseek_llm import DEFAULT_MODEL, make_complete  # noqa: E402
from agent.public_sanitize import sanitize_public_artifact  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "continual-qa.llm-control-flow.json"


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key-file", default=None)
    args = ap.parse_args()

    episodes = load_episodes(args.episodes)
    complete = make_complete(model=args.model, api_key_file=args.api_key_file)

    runs = []
    for i in range(args.runs):
        cf = control_flow_report(episodes, LLMController(complete=complete))
        runs.append({
            "run": i + 1,
            "substrateAccuracy": cf["substrateAccuracy"],
            "endToEndAccuracy": cf["endToEndAccuracy"],
            "controlFlowGap": cf["controlFlowGap"],
            "routingErrorCount": len(cf["routingErrors"]),
            "routingErrors": cf["routingErrors"],
        })
        print(f"run {i + 1}/{args.runs}: end-to-end={cf['endToEndAccuracy']} gap={cf['controlFlowGap']} "
              f"errors={len(cf['routingErrors'])}")

    e2e = [r["endToEndAccuracy"] for r in runs]
    gaps = [r["controlFlowGap"] for r in runs]
    report = {
        "schema": "sophia.continual_qa_llm_control_flow.v1",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "model": args.model,
        "provider": "deepseek",
        "episodes": [e.id for e in episodes],
        "runs": runs,
        "summary": {
            "substrateAccuracy": runs[0]["substrateAccuracy"] if runs else None,
            "endToEndAccuracyMean": _mean(e2e),
            "endToEndAccuracyRange": [min(e2e), max(e2e)] if e2e else None,
            "controlFlowGapMean": _mean(gaps),
            "controlFlowGapRange": [min(gaps), max(gaps)] if gaps else None,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
