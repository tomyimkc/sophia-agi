#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Continuous eval for the Sophia agent harness.

Runs a suite of goals through the harness and reports pass-rate, failure-class
histogram, cost, and latency, so every agent/model change is measurable. Offline
by default (mock provider); point --provider at any real model to grade it.

    python tools/eval_agent.py --provider mock
    python tools/eval_agent.py suites/agent_smoke.json --provider glm:glm-5.2 --out eval/results/agent.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import skills as skill_lib  # noqa: E402
from agent.harness import AgentTask, FAILURE_CLASSES, gate_verifier, run_agent  # noqa: E402
from agent.model import default_client  # noqa: E402

# A plumbing smoke suite: goals chosen so the OFFLINE mock provider can pass the
# gate (no attribution traps the mock cannot actually resolve). Point --provider
# at a real model to grade answer quality on harder, trap-bearing suites.
DEFAULT_SUITE: list[dict[str, Any]] = [
    {"id": "advisor_decision", "goal": "Should we launch Sophia on Hacker News this week?", "mode": "advisor", "mustInclude": ["Decision"]},
    {"id": "rag_summary", "goal": "Summarize why source discipline and provenance matter for an AGI.", "mode": "advisor", "skill": "research-rag", "mustInclude": ["Decision"]},
    {"id": "repo_next_step", "goal": "What should we validate before the next release?", "mode": "repo", "mustInclude": ["Decision"]},
]


def combined_verifier(must_include: list[str]):
    """Pass only if the gate passes AND every required token is present."""

    def _verify(text: str, task: AgentTask, step: dict) -> dict:
        gate = gate_verifier(text, task, step)
        lowered = text.lower()
        missing = [kw for kw in must_include if kw.lower() not in lowered]
        passed = gate["passed"] and not missing
        reasons = list(gate.get("reasons", [])) + [f"missing: {kw}" for kw in missing]
        return {"passed": passed, "reasons": reasons, "detail": {"gate": gate["detail"], "missing": missing}}

    return _verify


def run_suite(suite: list[dict[str, Any]], *, provider: str | None, max_retries: int) -> dict[str, Any]:
    client = default_client(provider)
    histogram = {cls: 0 for cls in FAILURE_CLASSES}
    results = []
    total_cost = 0.0
    total_latency = 0.0
    passed = 0
    for case in suite:
        skill = skill_lib.get(case["skill"]) if case.get("skill") else None
        verifier = combined_verifier(case.get("mustInclude", [])) if case.get("mustInclude") else gate_verifier
        task = AgentTask(goal=case["goal"], mode=case.get("mode", "advisor"), task_id=f"eval-{case['id']}", skill=skill)
        outcome = run_agent(task, client=client, verifier=verifier, max_retries=max_retries)
        total_cost += outcome.cost_usd
        total_latency += outcome.latency_sec
        if outcome.ok:
            passed += 1
        for step in outcome.steps:
            if step.failure_class:
                histogram[step.failure_class] = histogram.get(step.failure_class, 0) + 1
        results.append({
            "id": case["id"], "ok": outcome.ok, "failures": outcome.failures,
            "costUsd": round(outcome.cost_usd, 6), "latencySec": round(outcome.latency_sec, 3),
        })
    total = len(suite)
    return {
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "provider": provider or "auto",
        "caseCount": total,
        "passed": passed,
        "passRate": round(passed / total, 4) if total else 0.0,
        "failureHistogram": {k: v for k, v in histogram.items() if v},
        "meanCostUsd": round(total_cost / total, 6) if total else 0.0,
        "meanLatencySec": round(total_latency / total, 3) if total else 0.0,
        "totalCostUsd": round(total_cost, 6),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval the Sophia agent harness")
    parser.add_argument("suite", nargs="?", type=Path, default=None, help="suite JSON (defaults to a built-in smoke suite)")
    parser.add_argument("--provider", default="mock", help="model provider/preset (default mock for offline smoke)")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    suite = json.loads(args.suite.read_text(encoding="utf-8")) if args.suite else DEFAULT_SUITE
    report = run_suite(suite, provider=args.provider, max_retries=args.max_retries)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
