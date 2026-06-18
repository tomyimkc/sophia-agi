#!/usr/bin/env python3
"""Sophia AGI agent — three paths: advisor, repo, life.

Usage:
  python tools/sophia_agent.py advisor "Should I launch on HN this week?"
  python tools/sophia_agent.py repo "What should I do next?"
  python tools/sophia_agent.py repo "Export and upload corpus" --approve
  python tools/sophia_agent.py life "Should I take this job offer?"
  python tools/sophia_agent.py tools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response
from agent.llm import complete
from agent.memory import log_decision, recent_decisions
from agent.prompts import MODE_PROMPTS
from agent.retrieval import format_context, retrieve
from agent.tools import catalog_text, parse_tool_requests, run_tools

MODES = ("advisor", "repo", "life")


def build_user_prompt(mode: str, question: str) -> str:
    chunks = retrieve(question, top_k=8)
    context = format_context(chunks)
    memory = recent_decisions(limit=3)
    memory_text = json.dumps(memory, ensure_ascii=False, indent=2) if memory else "[]"
    extra = ""
    if mode == "repo":
        extra = f"\n\n## Repo tools\n{catalog_text()}\n"
    return (
        f"## User question\n{question}\n\n"
        f"## Retrieved sources\n{context}\n\n"
        f"## Recent decisions (memory)\n{memory_text}\n"
        f"{extra}\n"
        "Respond with: Analysis → Recommendation/Decision → cited source paths → 中文摘要"
    )


def run_mode(mode: str, question: str, *, approve: bool, execute: bool) -> int:
    if mode not in MODES:
        print(f"Unknown mode: {mode}. Choose: {', '.join(MODES)}")
        return 1

    print(f"[Sophia / {mode}] Retrieving sources...")
    chunks = retrieve(question, top_k=8)
    for chunk in chunks[:5]:
        print(f"  - {chunk.path} ({chunk.score:.2f})")

    print(f"[Sophia / {mode}] Thinking...")
    answer = complete(MODE_PROMPTS[mode], build_user_prompt(mode, question))
    gate = check_response(answer, mode=mode)

    print("\n" + "=" * 60 + "\n")
    print(answer)
    print("\n" + "=" * 60)
    print(f"\n[Gate] passed={gate['passed']} warnings={gate.get('warnings', [])}")

    tools_run: list[str] = []
    if mode == "repo" and execute:
        requested = parse_tool_requests(answer)
        if not requested:
            print("[Tools] No tool JSON block in response.")
        else:
            print(f"[Tools] Requested: {requested} (approve={approve})")
            results = run_tools(requested, approved=approve)
            tools_run = requested
            for result in results:
                status = "OK" if result.get("ok") else "FAIL"
                print(f"  {status} {result['tool']}")
                if result.get("stdout"):
                    print(result["stdout"][-500:])

    log_decision(
        mode=mode,
        question=question,
        answer=answer,
        sources=[c.path for c in chunks],
        gate=gate,
        tools_run=tools_run,
    )
    print(f"\n[Memory] Logged to agent/memory/decisions.jsonl")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sophia AGI agent (advisor | repo | life)")
    parser.add_argument("mode", choices=[*MODES, "tools"], help="Agent path or 'tools' to list repo tools")
    parser.add_argument("question", nargs="?", default="", help="Your question or decision prompt")
    parser.add_argument("--approve", action="store_true", help="Approve repo tool execution (repo mode)")
    parser.add_argument("--execute", action="store_true", help="Parse and run tools from repo agent response")
    args = parser.parse_args()

    if args.mode == "tools":
        print(catalog_text())
        return 0

    if not args.question:
        parser.error("question is required unless mode is 'tools'")

    return run_mode(args.mode, args.question, approve=args.approve, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())