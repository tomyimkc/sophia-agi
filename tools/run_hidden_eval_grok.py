#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run a private Sophia hidden evaluation pack against Grok CLI.

Full prompts and responses stay in private/hidden-evals/. The public report is
sanitized: no prompts, no materials, no answer text, and no rubric details.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hidden_eval_protocol import load_json, score_pack, validate_pack

ROOT = Path(__file__).resolve().parents[1]


def build_prompt(case: dict[str, Any]) -> str:
    materials = case.get("materials", [])
    material_text = "\n".join(f"- {item}" for item in materials) if materials else "(none)"
    return f"""You are Sophia AGI running a sealed hidden evaluation task.

Rules:
- Answer the task directly.
- Do not use tools, web search, filesystem access, shell commands, MCP, or repo inspection.
- Use source discipline when the task involves philosophy, psychology, history, or religion.
- Do not claim certainty beyond the evidence.
- For coding/planning/tool-use tasks, give concrete steps or code-level fixes.
- End with a short Decision section and 中文摘要.
- Produce the answer now; do not ask clarifying questions.

Hidden case id: {case["id"]}
Domain: {case["domain"]}

Task:
{case["prompt"]}

Materials:
{material_text}
"""


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def run_grok(prompt: str, *, timeout_sec: int) -> dict[str, Any]:
    command = [
        "grok",
        "-p",
        prompt,
        "--cwd",
        str(ROOT),
        "--output-format",
        "plain",
        "--max-turns",
        "8",
        "--no-plan",
        "--no-subagents",
        "--disable-web-search",
        "--system-prompt-override",
        "You are a direct benchmark respondent. Do not call tools. Do not inspect files. Do not plan. Answer the prompt from the text provided.",
    ]
    started = time.time()
    proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec, check=False)
    elapsed = round(time.time() - started, 2)
    return {
        "returncode": proc.returncode,
        "elapsedSec": elapsed,
        "stdout": strip_ansi(proc.stdout).strip(),
        "stderr": strip_ansi(proc.stderr).strip(),
    }


def sanitized_report(pack: dict[str, Any], private_report: dict[str, Any], response_payload: dict[str, Any]) -> dict[str, Any]:
    domains: dict[str, dict[str, Any]] = {}
    for result in private_report["results"]:
        item = domains.setdefault(result["domain"], {"passed": 0, "total": 0, "score": 0.0, "maxScore": 0.0})
        item["total"] += 1
        item["passed"] += 1 if result["passed"] else 0
        item["score"] += float(result["score"])
        item["maxScore"] += float(result["maxPoints"])

    for item in domains.values():
        item["scorePct"] = round((item["score"] / item["maxScore"]) * 100, 2) if item["maxScore"] else 0

    return {
        "packId": pack["packId"],
        "model": response_payload["model"],
        "runAt": response_payload["date"],
        "visibility": "public-aggregate-no-prompts",
        "hiddenStatus": "used-prepared-hidden-pack",
        "caseCount": len(pack["cases"]),
        "domains": sorted(domains),
        "passed": private_report["passed"],
        "totalCases": private_report["totalCases"],
        "score": private_report["score"],
        "maxScore": private_report["maxScore"],
        "scorePct": private_report["scorePct"],
        "scoreMethod": (
            "automatic keyword/avoidance screen with partial credit; "
            "manual rubric remains in the private reviewer report"
        ),
        "domainResults": domains,
        "caseResults": [
            {
                "id": result["id"],
                "domain": result["domain"],
                "passed": result["passed"],
                "score": result["score"],
                "maxPoints": result["maxPoints"],
                "passedChecks": result.get("passedChecks", 0),
                "totalChecks": result.get("totalChecks", 0),
            }
            for result in private_report["results"]
        ],
        "notes": [
            "Prepared pack is now spent; do not reuse as fresh hidden evidence.",
            "Prompt text, materials, responses, and detailed rubric remain private.",
            "Use a new third-party hidden pack for stronger AGI-candidate claims.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run private hidden eval pack with Grok CLI")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--responses-out", type=Path, required=True)
    parser.add_argument("--private-report-out", type=Path, required=True)
    parser.add_argument("--public-report-out", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--model-label", default="grok-cli")
    args = parser.parse_args()

    pack = load_json(args.pack)
    errors = validate_pack(pack)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, ensure_ascii=False))
        return 1

    responses: dict[str, str] = {}
    logs: dict[str, Any] = {}
    for index, case in enumerate(pack["cases"], 1):
        print(f"[{index}/{len(pack['cases'])}] {case['id']} ({case['domain']})")
        result = run_grok(build_prompt(case), timeout_sec=args.timeout_sec)
        logs[case["id"]] = {
            "returncode": result["returncode"],
            "elapsedSec": result["elapsedSec"],
            "stderrTail": result["stderr"][-4000:],
        }
        responses[case["id"]] = result["stdout"]
        if result["returncode"] != 0:
            print(f"  Grok failed with return code {result['returncode']}")

    response_payload = {
        "packId": pack["packId"],
        "model": args.model_label,
        "date": datetime.now().isoformat(timespec="seconds"),
        "responses": responses,
        "logs": logs,
    }
    private_report = score_pack(pack, response_payload)
    public_report = sanitized_report(pack, private_report, response_payload)

    for path, payload in (
        (args.responses_out, response_payload),
        (args.private_report_out, private_report),
        (args.public_report_out, public_report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path}")

    print(json.dumps(public_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
