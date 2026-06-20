#!/usr/bin/env python3
"""Run a private hidden evaluation pack through the full Sophia pipeline.

This runner exercises retrieval, prompt discipline, gate checks, one bounded
repair attempt, operational tool logs, and append-only learning memory diffs.
Hidden prompts/responses stay in private/hidden-evals/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from agent.gate import check_response  # noqa: E402
from agent.coding_council import format_coding_council, route_coding_council  # noqa: E402
from agent.llm import complete  # noqa: E402
from agent.prompts import MODE_PROMPTS, MODE_PROMPTS_NO_COUNCIL  # noqa: E402
from agent.retrieval import format_context, retrieve  # noqa: E402
from agent.rubric_review import build_rubric_review, format_rubric_review  # noqa: E402
from agent.web_evidence import format_evidence_context, gather_evidence  # noqa: E402
from hidden_eval_protocol import load_json, score_pack, validate_pack  # noqa: E402

MEMORY_FILE = ROOT / "agent" / "memory" / "hidden_eval_learning.jsonl"
DEFAULT_GROK_CWD = ROOT / "private" / "hidden-evals" / ".grok-isolated-cwd"
PROTECTED_KNOWLEDGE_FILES = [
    ROOT / "data" / "attributions.json",
    ROOT / "data" / "traditions.json",
    ROOT / "data" / "religion_concepts.json",
    ROOT / "data" / "psychology_concepts.json",
    ROOT / "data" / "history_events.json",
]


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str | None]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED_KNOWLEDGE_FILES}


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def task_mode(case: dict[str, Any]) -> str:
    return "repo" if case["domain"] in {"tool_use", "coding", "planning"} else "advisor"


def format_operational_evidence(
    *,
    tool_log: dict[str, Any] | None = None,
    memory_diff: dict[str, Any] | None = None,
    learning_probe: dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    if tool_log and tool_log.get("commands"):
        parts.append("### Tool log evidence")
        for command in tool_log["commands"]:
            parts.append(
                "- "
                f"`{command.get('cmd', '')}` returned {command.get('returncode')} "
                f"for {command.get('purpose', 'tool evidence')}."
            )
    if memory_diff:
        parts.append("### Append-only memory evidence")
        parts.append(
            "- "
            f"Memory file: `{memory_diff.get('memoryFile', '')}`; "
            f"appended={memory_diff.get('appended')}; "
            f"old protected records changed={memory_diff.get('oldHashChanged')}."
        )
        if memory_diff.get("entryRecordId"):
            parts.append(f"- New append-only record id: `{memory_diff['entryRecordId']}`.")
    if learning_probe:
        parts.append("### Learning probe evidence")
        if learning_probe.get("preTest"):
            parts.append(f"- Pre-test backend return code: {learning_probe['preTest'].get('returncode')}.")
        if learning_probe.get("postTest"):
            parts.append(f"- Post-test backend return code: {learning_probe['postTest'].get('returncode')}.")
        if "oldRecordsUnchanged" in learning_probe:
            parts.append(f"- Old records unchanged: {learning_probe.get('oldRecordsUnchanged')}.")
    return "\n".join(parts)


def build_user_prompt(
    case: dict[str, Any],
    context: str,
    *,
    repair: dict[str, Any] | None = None,
    coding_council: str = "",
    evidence_context: str = "",
    operational_evidence: str = "",
    rubric_review_context: str = "",
) -> str:
    materials = "\n".join(f"- {item}" for item in case.get("materials", [])) or "(none)"
    repair_text = ""
    if repair:
        repair_text = (
            "\n\n## Repair request\n"
            "Your previous answer missed these public check labels only. "
            "Do not ask for the hidden answer key; revise using the task and materials.\n"
            f"{json.dumps(repair, ensure_ascii=False, indent=2)}\n"
        )
    return f"""## Hidden evaluation task
Case ID: {case["id"]}
Domain: {case["domain"]}

Task:
{case["prompt"]}

Materials:
{materials}

## Retrieved Sophia context
{context}

{evidence_context}

{coding_council}

## Operational evidence already collected by runner
{operational_evidence or "(No operational evidence required for this case.)"}

{rubric_review_context}

## Required response contract
- Use explicit source discipline / provenance-aware wording when relevant.
- Cite exact local/web evidence labels when you rely on retrieved or online sources.
- Include a Decision section.
- Include a Rubric Evidence Map that maps each required item to a sentence, source, command log, memory diff, or uncertainty note.
- Include 中文摘要, but keep it to 1-3 short lines unless the task explicitly asks for a Chinese-first answer.
- For coding tasks, name the coding council seats, give patch-level or command-level specifics, and review tests/security/performance/edge cases.
- For tool-use tasks, cite the actual command/tool logs above and explain what they prove.
- For learning tasks, cite the append-only memory diff above and what old knowledge did not change.
- Before finalizing, run a judge pass: would this satisfy every rubric item and Sophia gate requirement?
{repair_text}
"""


def run_grok_direct(system: str, user: str, *, timeout_sec: int, grok_cwd: Path | None = None) -> dict[str, Any]:
    prompt = f"{system}\n\n{user}"
    run_cwd = (grok_cwd or DEFAULT_GROK_CWD).resolve()
    run_cwd.mkdir(parents=True, exist_ok=True)
    prompt_file: Path | None = None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=run_cwd,
        prefix="sophia-hidden-prompt-",
        suffix=".md",
        delete=False,
    ) as handle:
        handle.write(prompt)
        prompt_file = Path(handle.name)
    command = [
        "grok",
        "--prompt-file",
        str(prompt_file),
        "--cwd",
        str(run_cwd),
        "--output-format",
        "plain",
        "--max-turns",
        "8",
        "--no-memory",
        "--no-plan",
        "--no-subagents",
        "--disable-web-search",
        "--verbatim",
        "--system-prompt-override",
        "You are Sophia's hidden benchmark answerer. Do not call tools. Answer directly from the prompt and provided context.",
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=run_cwd,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "backend": "grok-cli",
            "returncode": 124,
            "elapsedSec": round(time.time() - started, 2),
            "answer": strip_ansi(exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderrTail": strip_ansi(exc.stderr or "").strip()[-4000:] if isinstance(exc.stderr, str) else "timeout",
            "timedOut": True,
            "runCwd": str(run_cwd),
        }
    except FileNotFoundError as exc:
        return {
            "backend": "grok-cli",
            "returncode": 127,
            "elapsedSec": round(time.time() - started, 2),
            "answer": "",
            "stderrTail": repr(exc),
            "missingExecutable": True,
            "runCwd": str(run_cwd),
        }
    finally:
        if prompt_file and prompt_file.exists():
            prompt_file.unlink()
    return {
        "backend": "grok-cli",
        "returncode": proc.returncode,
        "elapsedSec": round(time.time() - started, 2),
        "answer": strip_ansi(proc.stdout).strip(),
        "stderrTail": strip_ansi(proc.stderr).strip()[-4000:],
        "runCwd": str(run_cwd),
    }


def run_deepseek(system: str, user: str, *, timeout_sec: int) -> dict[str, Any]:
    started = time.time()
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return {
            "backend": "deepseek",
            "returncode": 1,
            "elapsedSec": round(time.time() - started, 2),
            "answer": "",
            "stderrTail": "Set DEEPSEEK_API_KEY in the environment or pass --deepseek-api-key-stdin.",
        }
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "temperature": float(os.environ.get("DEEPSEEK_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("DEEPSEEK_MAX_TOKENS", "2400")),
    }
    if os.environ.get("DEEPSEEK_REASONING_EFFORT"):
        payload["reasoning_effort"] = os.environ["DEEPSEEK_REASONING_EFFORT"]
    if os.environ.get("DEEPSEEK_THINKING"):
        payload["thinking"] = {"type": os.environ["DEEPSEEK_THINKING"]}
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-4000:]
        return {
            "backend": "deepseek",
            "model": model,
            "returncode": exc.code,
            "elapsedSec": round(time.time() - started, 2),
            "answer": "",
            "stderrTail": body,
            "httpStatus": exc.code,
        }
    except TimeoutError as exc:
        return {
            "backend": "deepseek",
            "model": model,
            "returncode": 124,
            "elapsedSec": round(time.time() - started, 2),
            "answer": "",
            "stderrTail": repr(exc),
            "timedOut": True,
        }
    except urllib.error.URLError as exc:
        return {
            "backend": "deepseek",
            "model": model,
            "returncode": 1,
            "elapsedSec": round(time.time() - started, 2),
            "answer": "",
            "stderrTail": repr(exc),
        }
    except Exception as exc:  # pragma: no cover - external API edge cases
        return {
            "backend": "deepseek",
            "model": model,
            "returncode": 1,
            "elapsedSec": round(time.time() - started, 2),
            "answer": "",
            "stderrTail": repr(exc),
        }
    try:
        data = json.loads(raw)
        answer = data["choices"][0]["message"].get("content") or ""
    except Exception as exc:
        return {
            "backend": "deepseek",
            "model": model,
            "returncode": 1,
            "elapsedSec": round(time.time() - started, 2),
            "answer": "",
            "stderrTail": f"Could not parse DeepSeek response: {exc!r}; body tail: {raw[-2000:]}",
        }
    return {
        "backend": "deepseek",
        "model": model,
        "returncode": 0,
        "elapsedSec": round(time.time() - started, 2),
        "answer": answer.strip(),
        "stderrTail": "",
    }


def call_model(
    system: str,
    user: str,
    *,
    backend: str,
    timeout_sec: int,
    grok_cwd: Path | None = None,
) -> dict[str, Any]:
    started = time.time()
    if backend == "anthropic":
        try:
            answer = complete(system, user)
            return {
                "backend": "anthropic",
                "returncode": 0,
                "elapsedSec": round(time.time() - started, 2),
                "answer": answer,
                "stderrTail": "",
            }
        except Exception as exc:  # pragma: no cover - depends on local API config
            return {
                "backend": "anthropic",
                "returncode": 1,
                "elapsedSec": round(time.time() - started, 2),
                "answer": "",
                "stderrTail": repr(exc),
            }
    if backend == "grok":
        return run_grok_direct(system, user, timeout_sec=timeout_sec, grok_cwd=grok_cwd)
    if backend == "deepseek":
        return run_deepseek(system, user, timeout_sec=timeout_sec)
    raise ValueError(f"unknown backend: {backend}")


def backend_preflight(*, backend: str, timeout_sec: int, grok_cwd: Path | None = None) -> dict[str, Any]:
    marker = "SOPHIA_PREFLIGHT_OK"
    result = call_model(
        "You are a benchmark backend preflight responder.",
        (
            "This is a preflight for a hidden evaluation runner. "
            "Return the marker, a Decision line, and a short 中文摘要. "
            f"The marker is: {marker}"
        ),
        backend=backend,
        timeout_sec=timeout_sec,
        grok_cwd=grok_cwd,
    )
    answer = result.get("answer", "")
    has_zh = "中文摘要" in answer or bool(re.search(r"[\u4e00-\u9fff]", answer))
    ok = result.get("returncode") == 0 and marker in answer and "Decision" in answer and has_zh
    return {
        **{k: v for k, v in result.items() if k != "answer"},
        "ok": ok,
        "answerPreview": answer[:200],
        "expectedMarker": marker,
    }


def run_operational_tools(case: dict[str, Any]) -> dict[str, Any]:
    if not case.get("requiresToolLog"):
        return {}
    commands = [
        {"argv": ["git", "status", "--short"], "purpose": "check repo working tree state"},
        {"argv": ["find", "agi-proof", "-maxdepth", "3", "-type", "f"], "purpose": "inspect proof/evidence files"},
        {
            "argv": ["python", "-m", "json.tool", "agi-proof/evidence-manifest.json"],
            "purpose": "validate proof manifest JSON",
        },
    ]
    results = []
    for item in commands:
        proc = subprocess.run(
            item["argv"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        results.append(
            {
                "cmd": " ".join(item["argv"]),
                "purpose": item["purpose"],
                "returncode": proc.returncode,
                "stdoutTail": proc.stdout[-1000:],
                "stderrTail": proc.stderr[-1000:],
            }
        )
    return {"commands": results}


def append_learning_memory(case: dict[str, Any]) -> dict[str, Any]:
    if not case.get("requiresMemoryDiff"):
        return {}
    protected_before = protected_hashes()
    old_hash = sha256_file(MEMORY_FILE)
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "caseId": case["id"],
        "recordId": f"hidden_eval_{case['id']}",
        "mode": "append-only",
        "materialsHash": hashlib.sha256(json.dumps(case.get("materials", []), sort_keys=True).encode()).hexdigest(),
        "oldKnowledgePolicy": "do not overwrite existing religion/domain records",
    }
    with MEMORY_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    new_hash = sha256_file(MEMORY_FILE)
    protected_after = protected_hashes()
    return {
        "memoryFile": str(MEMORY_FILE.relative_to(ROOT)),
        "oldHash": old_hash,
        "newHash": new_hash,
        "appended": old_hash != new_hash,
        "protectedKnowledgeBefore": protected_before,
        "protectedKnowledgeAfter": protected_after,
        "oldHashChanged": protected_before != protected_after,
        "entryRecordId": entry["recordId"],
    }


def run_learning_probe(
    case: dict[str, Any],
    context: str,
    system: str,
    *,
    backend: str,
    timeout_sec: int,
    grok_cwd: Path | None = None,
) -> dict[str, Any]:
    if not case.get("requiresMemoryDiff"):
        return {}
    learning = case.get("learningProtocol", {})
    pre_prompt = learning.get("preTestPrompt") or (
        f"Pre-test before append-only memory update. Answer without using the new memory entry.\n\n{case['prompt']}"
    )
    post_prompt = learning.get("postTestPrompt") or (
        f"Post-test after append-only memory update. Use the new appended memory if relevant.\n\n{case['prompt']}"
    )
    pre = call_model(
        system,
        build_user_prompt({**case, "prompt": pre_prompt}, context),
        backend=backend,
        timeout_sec=timeout_sec,
        grok_cwd=grok_cwd,
    )
    if pre.get("returncode") != 0 or not pre.get("answer", "").strip():
        return {
            "preTest": {k: v for k, v in pre.items() if k != "answer"},
            "preAnswer": pre.get("answer", ""),
            "memoryDiff": {},
            "postTest": {},
            "postAnswer": "",
            "oldRecordsUnchanged": False,
            "skippedAppend": True,
            "skipReason": "pre-test failed or returned empty response",
        }
    memory_diff = append_learning_memory(case)
    post = call_model(
        system,
        build_user_prompt({**case, "prompt": post_prompt}, context),
        backend=backend,
        timeout_sec=timeout_sec,
        grok_cwd=grok_cwd,
    )
    return {
        "preTest": {k: v for k, v in pre.items() if k != "answer"},
        "preAnswer": pre.get("answer", ""),
        "memoryDiff": memory_diff,
        "postTest": {k: v for k, v in post.items() if k != "answer"},
        "postAnswer": post.get("answer", ""),
        "oldRecordsUnchanged": not memory_diff.get("oldHashChanged", True),
    }


def repair_payload(score_result: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "failedInclude": score_result.get("failedInclude", []),
        "failedAvoid": score_result.get("failedAvoid", []),
        "missedRubric": score_result.get("missedRubric", []),
        "operationalFailures": score_result.get("operationalFailures", []),
        "gateWarnings": gate.get("warnings", []),
        "gateViolations": gate.get("violations", []),
    }


def review_payload(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "strictPassReady": review.get("strictPassReady", False),
        "missing": review.get("missing", []),
        "revisionInstructions": review.get("revisionInstructions", []),
        "operationalEvidence": review.get("operationalEvidence", {}),
        "sourceEvidence": review.get("sourceEvidence", {}),
    }


def should_attempt_repair(*, enabled: bool, first: dict[str, Any], provisional: dict[str, Any], gate: dict[str, Any]) -> bool:
    return bool(
        enabled
        and first.get("returncode") == 0
        and (not provisional.get("passed") or not gate.get("passed"))
    )


def council_public_summary(payload: dict[str, Any]) -> dict[str, Any]:
    routes = payload.get("codingCouncilRoutes", {})
    cases: dict[str, Any] = {}
    for case_id, route in routes.items():
        seats = []
        for key in ("languageSeats", "roleSeats", "platformSeats", "specialistSeats", "improvementSeats"):
            for seat in route.get(key, []):
                if isinstance(seat, dict) and seat.get("displayName"):
                    seats.append(seat["displayName"])
        cases[case_id] = {
            "seatCount": len(seats),
            "seats": seats,
        }
    return {
        "engineeringCouncilCasesRouted": len(cases),
        "cases": cases,
    }


def sanitized_report(pack: dict[str, Any], private_report: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    domains: dict[str, dict[str, Any]] = {}
    for result in private_report["results"]:
        item = domains.setdefault(result["domain"], {"passed": 0, "total": 0, "score": 0.0, "maxScore": 0.0})
        item["total"] += 1
        item["passed"] += 1 if result["passed"] else 0
        item["score"] += float(result["score"])
        item["maxScore"] += float(result["maxPoints"])
    for item in domains.values():
        item["scorePct"] = round((item["score"] / item["maxScore"]) * 100, 2) if item["maxScore"] else 0
    response_map = payload.get("responses", {})
    model_logs = payload.get("logs", {})
    backend_failures = {
        case_id: {
            "returncode": log.get("returncode"),
            "timedOut": log.get("timedOut", False),
            "missingExecutable": log.get("missingExecutable", False),
        }
        for case_id, log in model_logs.items()
        if log.get("returncode") not in (None, 0)
    }
    nonempty_answers = sum(1 for answer in response_map.values() if str(answer).strip())
    rubric_reviews = payload.get("rubricReviews", {})
    strict_ready_count = sum(1 for review in rubric_reviews.values() if review.get("strictPassReady"))
    manual_pending_count = sum(
        1
        for review in rubric_reviews.values()
        for item in review.get("semanticChecks", [])
        if item.get("status") in {"pending-manual-review", "needs-adjudication"}
    )
    web_evidence = payload.get("webEvidence", {})
    web_enabled_cases = sum(1 for item in web_evidence.values() if item.get("web", {}).get("online"))
    web_source_count = sum(len(item.get("web", {}).get("sources", [])) for item in web_evidence.values())
    return {
        "packId": pack["packId"],
        "model": payload["model"],
        "runAt": payload["date"],
        "visibility": "public-aggregate-no-prompts",
        "hiddenStatus": "used-hidden-pack",
        "caseCount": len(pack["cases"]),
        "domains": sorted(domains),
        "passed": private_report["passed"],
        "totalCases": private_report["totalCases"],
        "score": private_report["score"],
        "maxScore": private_report["maxScore"],
        "scorePct": private_report["scorePct"],
        "scoreMethod": (
            "alias/regex keyword screen plus operational tool/memory evidence; "
            "manual semantic judge review remains required for strong claims"
        ),
        "backendHealth": payload.get("backendHealth", {}),
        "responseHealth": {
            "nonemptyAnswers": nonempty_answers,
            "totalAnswers": len(response_map),
            "backendFailureCount": len(backend_failures),
            "backendFailures": backend_failures,
        },
        "repairAttempts": payload.get("repairAttempts", {}),
        "councilRouting": council_public_summary(payload),
        "rubricReviewHealth": {
            "strictReadyCount": strict_ready_count,
            "totalReviewed": len(rubric_reviews),
            "manualSemanticPendingOrAdjudicationCount": manual_pending_count,
            "casesWithMissingItems": {
                case_id: len(review.get("missing", []))
                for case_id, review in rubric_reviews.items()
                if review.get("missing")
            },
        },
        "webEvidenceHealth": {
            "enabledCases": web_enabled_cases,
            "webSourceCount": web_source_count,
            "note": "Online search is opt-in because hidden prompts should not be sent to third-party APIs without reviewer approval.",
        },
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
                "requiresManualReview": result.get("requiresManualReview", False),
                "manualReview": result.get("manualReview", "pending"),
                "missedRubricCount": len(result.get("missedRubric", [])),
                "rubricReviewMissingCount": len(rubric_reviews.get(result["id"], {}).get("missing", [])),
                "strictPassReadyByReview": rubric_reviews.get(result["id"], {}).get("strictPassReady", False),
            }
            for result in private_report["results"]
        ],
        "notes": [
            "Full prompts, responses, detailed rubrics, tool output, and memory diffs remain private.",
            "This runner exercises Sophia retrieval, gate checks, bounded repair, tool logs, and append-only memory evidence.",
            "Rubric review health is deterministic and does not replace two-pass human semantic review.",
            "Backend failures and empty model answers are recorded separately from semantic failures.",
            "Fresh third-party packs still require an external reviewer signature.",
        ],
    }


def manual_review_template(pack: dict[str, Any], private_report: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    results_by_id = {result["id"]: result for result in private_report["results"]}
    cases: dict[str, Any] = {}
    for case in pack["cases"]:
        scoring = case["scoring"]
        result = results_by_id[case["id"]]
        cases[case["id"]] = {
            "domain": case["domain"],
            "prompt": case.get("prompt", ""),
            "materials": case.get("materials", []),
            "response": payload.get("responses", {}).get(case["id"], ""),
            "sources": payload.get("sources", {}).get(case["id"], []),
            "webEvidence": payload.get("webEvidence", {}).get(case["id"], {}),
            "gate": payload.get("gates", {}).get(case["id"], {}),
            "rubricReview": payload.get("rubricReviews", {}).get(case["id"], {}),
            "codingCouncilRoute": payload.get("codingCouncilRoutes", {}).get(case["id"], {}),
            "automaticScore": {
                "passed": result.get("passed"),
                "score": result.get("score"),
                "maxPoints": result.get("maxPoints"),
                "passedChecks": result.get("passedChecks"),
                "totalChecks": result.get("totalChecks"),
            },
            "requiresManualReview": result.get("requiresManualReview", False),
            "manualReview": result.get("manualReview", "pending"),
            "failedInclude": result.get("failedInclude", []),
            "failedAvoid": result.get("failedAvoid", []),
            "operationalFailures": result.get("operationalFailures", []),
            "missedRubric": result.get("missedRubric", []),
            "rubric": scoring.get("rubric", []),
            "semanticChecks": {
                item.get("id", f"semantic_{index}"): {
                    "description": item.get("description", ""),
                    "reviewers": [
                        {"judge": "", "passed": None, "confidence": "", "notes": ""},
                        {"judge": "", "passed": None, "confidence": "", "notes": ""},
                    ],
                    "adjudication": {"judge": "", "passed": None, "notes": ""},
                }
                for index, item in enumerate(scoring.get("semanticChecks", []), 1)
                if isinstance(item, dict)
            },
            "overallHumanDecision": {
                "acceptedAsStrictPass": None,
                "judge": "",
                "notes": "",
            },
            "notes": "",
        }
    return {
        "packId": pack["packId"],
        "status": "pending-two-pass-human-review",
        "instructions": [
            "Two independent reviewers fill every semantic check.",
            "If reviewers disagree, a lead reviewer fills adjudication.",
            "Do not edit prompts, responses, or automatic score fields.",
            "Re-score with hidden_eval_protocol.py --manual-review after review.",
        ],
        "manualJudgements": cases,
    }


def failure_training_candidates(private_report: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for result in private_report.get("results", []):
        missed = result.get("missedRubric", [])
        if not missed:
            continue
        case_id = result["id"]
        candidates.append(
            {
                "caseId": case_id,
                "domain": result.get("domain"),
                "source": "hidden-eval-private-review",
                "status": "candidate-not-promoted",
                "failedAnswer": payload.get("responses", {}).get(case_id, ""),
                "missedRubric": missed,
                "repairInstruction": (
                    "Produce a repaired answer addressing every missed rubric item. "
                    "Keep hidden-eval data private and do not promote to public training until reviewed."
                ),
            }
        )
    return candidates


# ---------------------------------------------------------------------------
# Reusable single-case pipeline (shared by the hidden runner and the ablation
# runner). The default ablation is sophia-full, so main() behaviour is
# unchanged; run_ablation_sophia.py passes component-suppressed variants.
# ---------------------------------------------------------------------------

RAW_SYSTEM_PROMPT = (
    "You are a capable, knowledgeable assistant. Answer the task as accurately, "
    "completely, and helpfully as you can, showing brief reasoning where it helps. "
    "There are no special formatting requirements."
)

NEUTRAL_GATE: dict[str, Any] = {
    "passed": True,
    "warnings": [],
    "violations": [],
    "checks": [],
    "gateApplied": False,
}


@dataclass(frozen=True)
class RunConfig:
    """Backend/runtime configuration shared by every case in a run."""

    backend: str
    timeout_sec: int = 240
    grok_cwd: Path | None = None
    repair: bool = False
    online_evidence: bool = False
    web_provider: str = "off"
    web_search_top_k: int = 5
    local_evidence_top_k: int = 3


@dataclass(frozen=True)
class Ablation:
    """Which Sophia components are active for a case.

    Maps 1:1 onto the seven modes in agi-proof/baseline-ablation/README.md.
    """

    label: str = "sophia-full"
    raw_system: bool = False  # neutral RAW_SYSTEM_PROMPT instead of MODE_PROMPTS
    use_kb: bool = True  # retrieval context (local source records)
    use_evidence: bool = True  # local/web evidence gathering
    use_council: bool = True  # coding/figure council synthesis
    use_gate: bool = True  # post-generation epistemic gate
    use_memory: bool = True  # append-only learning probe + memory diff
    use_tools: bool = True  # operational tool logs
    allow_repair: bool = True  # bounded Sophia repair attempt


SOPHIA_FULL = Ablation()

ABLATION_MODES: dict[str, Ablation] = {
    "raw-model": Ablation(
        label="raw-model",
        raw_system=True,
        use_kb=False,
        use_evidence=False,
        use_council=False,
        use_gate=False,
        use_memory=False,
        use_tools=False,
        allow_repair=False,
    ),
    "raw-model-plus-tools": Ablation(
        label="raw-model-plus-tools",
        raw_system=True,
        use_kb=False,
        use_evidence=False,
        use_council=False,
        use_gate=False,
        use_memory=False,
        use_tools=True,
        allow_repair=False,
    ),
    "sophia-full": SOPHIA_FULL,
    "sophia-no-kb": Ablation(label="sophia-no-kb", use_kb=False, use_evidence=False),
    "sophia-no-gate": Ablation(label="sophia-no-gate", use_gate=False),
    "sophia-no-memory": Ablation(label="sophia-no-memory", use_memory=False),
    "sophia-no-council": Ablation(label="sophia-no-council", use_council=False),
}


def build_raw_user_prompt(case: dict[str, Any], *, operational_evidence: str = "") -> str:
    """Minimal task prompt with no Sophia source-discipline contract.

    Used by the raw-model baselines so the comparison does not leak Sophia
    discipline into the base model.
    """
    materials = "\n".join(f"- {item}" for item in case.get("materials", [])) or "(none)"
    tool_block = ""
    if operational_evidence:
        tool_block = f"\n\n## Tool output available to you\n{operational_evidence}\n"
    return f"""Task ({case["domain"]}):
{case["prompt"]}

Materials:
{materials}
{tool_block}"""


def run_case(
    case: dict[str, Any],
    pack_id: str,
    *,
    config: RunConfig,
    ablation: Ablation = SOPHIA_FULL,
) -> dict[str, Any]:
    """Run one case through the (optionally ablated) Sophia pipeline.

    Returns every per-case artifact both runners need so they can assemble an
    identical payload shape for score_pack/sanitized_report.
    """
    case_id = case["id"]

    if ablation.use_kb:
        chunks = retrieve(case["prompt"], top_k=8)
        context = format_context(chunks)
        case_sources = [chunk.path for chunk in chunks]
    else:
        context = ""
        case_sources = []

    if ablation.use_evidence:
        evidence = gather_evidence(
            case["prompt"],
            local_top_k=config.local_evidence_top_k,
            web_top_k=config.web_search_top_k,
            online=config.online_evidence,
            provider=config.web_provider,
        )
        evidence_context = format_evidence_context(evidence)
    else:
        evidence = {}
        evidence_context = ""

    mode = task_mode(case)
    if ablation.raw_system:
        system = RAW_SYSTEM_PROMPT
    elif not ablation.use_council:
        # Ablate council instructions at the prompt level too, not just the
        # structured coding route, so sophia-no-council genuinely removes
        # council-style multi-voice synthesis (incl. the religion-figure council).
        system = MODE_PROMPTS_NO_COUNCIL[mode]
    else:
        system = MODE_PROMPTS[mode]

    council_route: dict[str, Any] = {}
    council_context = ""
    if ablation.use_council and case["domain"] in {"coding", "tool_use", "planning", "learning"}:
        council_route = route_coding_council(case["prompt"], case.get("materials", []))
        council_context = format_coding_council(council_route)

    tool_log = run_operational_tools(case) if ablation.use_tools else {}

    learning_probe: dict[str, Any] = {}
    if ablation.use_memory:
        learning_probe = run_learning_probe(
            case,
            context,
            system,
            backend=config.backend,
            timeout_sec=config.timeout_sec,
            grok_cwd=config.grok_cwd,
        )
    memory_diff = learning_probe.get("memoryDiff", {})

    operational_evidence = format_operational_evidence(
        tool_log=tool_log,
        memory_diff=memory_diff,
        learning_probe=learning_probe,
    )

    if ablation.raw_system:
        user = build_raw_user_prompt(
            case,
            operational_evidence=operational_evidence if ablation.use_tools else "",
        )
    else:
        user = build_user_prompt(
            case,
            context,
            coding_council=council_context,
            evidence_context=evidence_context,
            operational_evidence=operational_evidence,
        )

    first = call_model(
        system,
        user,
        backend=config.backend,
        timeout_sec=config.timeout_sec,
        grok_cwd=config.grok_cwd,
    )
    answer = first["answer"]
    if ablation.use_gate:
        gate = check_response(answer, mode=mode, question=case["prompt"], domain=None)
    else:
        gate = dict(NEUTRAL_GATE)

    def _score_one(candidate: str) -> dict[str, Any]:
        return score_pack(
            {"packId": pack_id, "cases": [case]},
            {
                "responses": {case_id: candidate},
                "toolLogs": {case_id: tool_log},
                "memoryDiffs": {case_id: memory_diff},
            },
        )["results"][0]

    provisional = _score_one(answer)
    review = build_rubric_review(
        case,
        answer,
        provisional,
        gate,
        sources=case_sources,
        evidence=evidence,
        tool_log=tool_log,
        memory_diff=memory_diff,
    )

    repair_count = 0
    if should_attempt_repair(
        enabled=config.repair and ablation.allow_repair,
        first=first,
        provisional=provisional,
        gate=gate,
    ):
        repair_count = 1
        repair = {
            **repair_payload(provisional, gate),
            "rubricReview": review_payload(review),
        }
        second = call_model(
            system,
            build_user_prompt(
                case,
                context,
                repair=repair,
                coding_council=council_context,
                evidence_context=evidence_context,
                operational_evidence=operational_evidence,
                rubric_review_context=format_rubric_review(review),
            ),
            backend=config.backend,
            timeout_sec=config.timeout_sec,
            grok_cwd=config.grok_cwd,
        )
        if second["answer"].strip():
            answer = second["answer"]
            first["repair"] = second
            if ablation.use_gate:
                gate = check_response(answer, mode=mode, question=case["prompt"], domain=None)
            final_score = _score_one(answer)
            review = build_rubric_review(
                case,
                answer,
                final_score,
                gate,
                sources=case_sources,
                evidence=evidence,
                tool_log=tool_log,
                memory_diff=memory_diff,
            )

    model_log = {k: v for k, v in first.items() if k != "answer"}
    if learning_probe:
        model_log["learningProbe"] = learning_probe

    return {
        "answer": answer,
        "sources": case_sources,
        "gate": gate,
        "modelLog": model_log,
        "toolLog": tool_log,
        "memoryDiff": memory_diff,
        "repairAttempts": repair_count,
        "codingCouncilRoute": council_route,
        "webEvidence": evidence,
        "rubricReview": review,
        "returncode": first.get("returncode"),
        "elapsedSec": first.get("elapsedSec"),
        "ablation": ablation.label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hidden eval pack through Sophia pipeline")
    parser.add_argument("pack", type=Path)
    parser.add_argument(
        "--backend",
        choices=["anthropic", "grok", "deepseek"],
        default=os.environ.get("SOPHIA_HIDDEN_BACKEND", "grok"),
    )
    parser.add_argument("--responses-out", type=Path, required=True)
    parser.add_argument("--private-report-out", type=Path, required=True)
    parser.add_argument("--public-report-out", type=Path, required=True)
    parser.add_argument("--manual-review-out", type=Path)
    parser.add_argument("--failure-training-out", type=Path)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--preflight-timeout-sec", type=int, default=45)
    parser.add_argument(
        "--grok-cwd",
        type=Path,
        default=DEFAULT_GROK_CWD,
        help="Isolated cwd for Grok CLI calls; avoids project MCP/plugin startup during hidden answers.",
    )
    parser.add_argument("--model-label", default="sophia-full-hidden")
    parser.add_argument(
        "--deepseek-api-key-stdin",
        action="store_true",
        help="Read DeepSeek API key from stdin for this process only; avoids saving secrets to files.",
    )
    parser.add_argument("--repair", action="store_true", help="Allow one bounded repair attempt per case")
    parser.add_argument(
        "--web-evidence",
        action="store_true",
        help=(
            "Opt in to external web evidence search. This may send hidden prompt text "
            "to SOPHIA_WEB_SEARCH_PROVIDER, so use only with reviewer approval."
        ),
    )
    parser.add_argument(
        "--web-provider",
        choices=["off", "auto", "brave", "tavily", "serpapi"],
        default=os.environ.get("SOPHIA_WEB_SEARCH_PROVIDER", "off"),
    )
    parser.add_argument("--web-search-top-k", type=int, default=5)
    parser.add_argument("--local-evidence-top-k", type=int, default=3)
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Dangerous: run hidden cases without backend health check. Use only for smoke tests.",
    )
    args = parser.parse_args()

    if args.deepseek_api_key_stdin:
        print("[secret] waiting for DeepSeek API key on stdin", file=sys.stderr, flush=True)
        key = sys.stdin.readline().strip()
        if not key:
            print(json.dumps({"ok": False, "stage": "secret-input", "error": "empty DeepSeek API key"}, indent=2))
            return 1
        os.environ["DEEPSEEK_API_KEY"] = key

    pack = load_json(args.pack)
    errors = validate_pack(pack)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, ensure_ascii=False))
        return 1

    backend_health = {"ok": True, "skipped": True, "backend": args.backend}
    if not args.skip_preflight:
        print(f"[preflight] checking {args.backend} backend before exposing hidden pack")
        backend_health = backend_preflight(
            backend=args.backend,
            timeout_sec=args.preflight_timeout_sec,
            grok_cwd=args.grok_cwd,
        )
        if not backend_health.get("ok"):
            print(json.dumps({"ok": False, "stage": "backend-preflight", "backendHealth": backend_health}, indent=2, ensure_ascii=False))
            return 2

    responses: dict[str, str] = {}
    sources: dict[str, list[str]] = {}
    gates: dict[str, Any] = {}
    model_logs: dict[str, Any] = {}
    tool_logs: dict[str, Any] = {}
    memory_diffs: dict[str, Any] = {}
    repair_attempts: dict[str, int] = {}
    coding_council_routes: dict[str, Any] = {}
    web_evidence: dict[str, Any] = {}
    rubric_reviews: dict[str, Any] = {}

    config = RunConfig(
        backend=args.backend,
        timeout_sec=args.timeout_sec,
        grok_cwd=args.grok_cwd,
        repair=args.repair,
        online_evidence=args.web_evidence,
        web_provider=args.web_provider,
        web_search_top_k=args.web_search_top_k,
        local_evidence_top_k=args.local_evidence_top_k,
    )

    for index, case in enumerate(pack["cases"], 1):
        print(f"[{index}/{len(pack['cases'])}] {case['id']} ({case['domain']})", flush=True)
        result = run_case(case, pack["packId"], config=config, ablation=SOPHIA_FULL)
        responses[case["id"]] = result["answer"]
        sources[case["id"]] = result["sources"]
        gates[case["id"]] = result["gate"]
        model_logs[case["id"]] = result["modelLog"]
        tool_logs[case["id"]] = result["toolLog"]
        memory_diffs[case["id"]] = result["memoryDiff"]
        repair_attempts[case["id"]] = result["repairAttempts"]
        web_evidence[case["id"]] = result["webEvidence"]
        rubric_reviews[case["id"]] = result["rubricReview"]
        if result["codingCouncilRoute"]:
            coding_council_routes[case["id"]] = result["codingCouncilRoute"]
        if result["returncode"] not in (None, 0):
            print(f"  backend returned {result['returncode']}", flush=True)

    payload = {
        "packId": pack["packId"],
        "model": args.model_label,
        "backend": args.backend,
        "backendHealth": backend_health,
        "date": datetime.now().isoformat(timespec="seconds"),
        "responses": responses,
        "sources": sources,
        "gates": gates,
        "logs": model_logs,
        "toolLogs": tool_logs,
        "memoryDiffs": memory_diffs,
        "repairAttempts": repair_attempts,
        "codingCouncilRoutes": coding_council_routes,
        "webEvidence": web_evidence,
        "rubricReviews": rubric_reviews,
    }
    private_report = score_pack(pack, payload)
    public_report = sanitized_report(pack, private_report, payload)
    review_template = manual_review_template(pack, private_report, payload)

    outputs = [
        (args.responses_out, payload),
        (args.private_report_out, private_report),
        (args.public_report_out, public_report),
    ]
    if args.manual_review_out:
        outputs.append((args.manual_review_out, review_template))
    if args.failure_training_out:
        outputs.append((args.failure_training_out, failure_training_candidates(private_report, payload)))

    for path, data in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {path}")

    print(json.dumps(public_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
