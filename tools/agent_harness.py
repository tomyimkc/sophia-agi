#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CLI for the Sophia agent harness (plan -> act -> critic -> reflect/retry).

    # offline smoke (no creds): uses the mock model provider
    python tools/agent_harness.py run "Should we launch on HN this week?" --provider mock

    # real run against any provider (GLM-5.2 / local / anthropic / grok)
    SOPHIA_MODEL_PROVIDER=glm SOPHIA_MODEL=glm-5.2 ZHIPUAI_API_KEY=... \
      python tools/agent_harness.py run "Fix the failing auth test" --skill coding-debugging

    python tools/agent_harness.py skills          # list skills
    python tools/agent_harness.py models           # show resolved model config
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import skills as skill_lib  # noqa: E402
from agent.harness import AgentTask, run_agent  # noqa: E402
from agent.model import PRESETS, default_client, resolve_config  # noqa: E402


def cmd_run(args: argparse.Namespace) -> int:
    skill = None
    if args.skill:
        skill = skill_lib.get(args.skill)
        if skill is None:
            print(f"unknown skill: {args.skill}; available: {[s['name'] for s in skill_lib.list_skills()]}")
            return 1
    elif args.auto_skill:
        skill = skill_lib.select(args.goal)

    client = default_client(args.provider)
    task = AgentTask(goal=args.goal, mode=args.mode, task_id=args.task_id or "", skill=skill)
    result = run_agent(
        task,
        client=client,
        max_retries=args.max_retries,
        max_steps=args.max_steps,
        approve_tools=args.approve_tools,
        resume=args.resume,
    )
    if args.json:
        print(json.dumps({
            "taskId": result.task_id, "ok": result.ok, "failures": result.failures,
            "costUsd": round(result.cost_usd, 6), "latencySec": round(result.latency_sec, 3),
            "tracePath": result.trace_path,
            "steps": [{"id": s.step_id, "ok": s.ok, "attempts": s.attempts, "failureClass": s.failure_class} for s in result.steps],
        }, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 60)
        print(result.final_text)
        print("=" * 60)
        print(f"[harness] ok={result.ok} skill={(skill or {}).get('name')} "
              f"steps={len(result.steps)} cost=${result.cost_usd:.4f} latency={result.latency_sec:.2f}s")
        if result.failures:
            print(f"[harness] failures: {result.failures}")
        print(f"[harness] trace: {result.trace_path}")
    return 0 if result.ok else 2


def cmd_skills(args: argparse.Namespace) -> int:
    for item in skill_lib.list_skills():
        print(f"{item['name']:28s} {item['whenToUse']}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    print("Presets:", ", ".join(sorted(PRESETS)))
    cfg = resolve_config(args.provider)
    print("\nResolved primary config:")
    print(json.dumps({k: v for k, v in asdict(cfg).items() if k != "api_key_default"}, indent=2))
    print(f"api key present: {bool(cfg.resolved_key())}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sophia agent harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a goal through the agent loop")
    run.add_argument("goal")
    run.add_argument("--mode", choices=["advisor", "repo", "life"], default="advisor")
    run.add_argument("--provider", default=None, help="model provider/preset (e.g. mock, glm, glm:glm-5.2, ollama:llama3.1, anthropic)")
    run.add_argument("--skill", default=None, help="force a skill by name")
    run.add_argument("--auto-skill", action="store_true", help="auto-select a skill from the goal")
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--max-steps", type=int, default=4)
    run.add_argument("--approve-tools", action="store_true", help="allow repo tool execution")
    run.add_argument("--resume", action="store_true", help="resume from a prior run trace")
    run.add_argument("--task-id", default=None)
    run.add_argument("--json", action="store_true")
    run.set_defaults(func=cmd_run)

    sk = sub.add_parser("skills", help="list available skills")
    sk.set_defaults(func=cmd_skills)

    md = sub.add_parser("models", help="show model presets and resolved config")
    md.add_argument("--provider", default=None)
    md.set_defaults(func=cmd_models)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
