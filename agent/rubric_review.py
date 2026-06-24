# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic rubric review for Sophia hidden-eval answers."""

from __future__ import annotations

import re
from typing import Any


def build_rubric_review(
    case: dict[str, Any],
    response: str,
    score_result: dict[str, Any],
    gate: dict[str, Any],
    *,
    sources: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    tool_log: dict[str, Any] | None = None,
    memory_diff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scoring = case.get("scoring", {})
    required = scoring.get("mustInclude", [])
    forbidden = scoring.get("mustAvoid", [])
    semantic = scoring.get("semanticChecks", [])
    aliases = scoring.get("aliases", {})

    required_items = [_review_item(response, item, aliases=aliases, required=True) for item in required]
    forbidden_items = [_review_item(response, item, aliases=aliases, required=False) for item in forbidden]
    semantic_items = _semantic_items(score_result, semantic)
    evidence_items = _evidence_items(sources=sources or [], evidence=evidence or {})
    operational = _operational_review(case, tool_log=tool_log, memory_diff=memory_diff)
    missing = _missing_items(required_items, forbidden_items, semantic_items, operational, gate)

    strict_ready = bool(
        response.strip()
        and score_result.get("passed")
        and gate.get("passed")
        and not missing
        and operational.get("passed", True)
    )
    return {
        "caseId": case.get("id"),
        "domain": case.get("domain"),
        "strictPassReady": strict_ready,
        "answerNonempty": bool(response.strip()),
        "requiredEvidence": required_items,
        "forbiddenClaims": forbidden_items,
        "semanticChecks": semantic_items,
        "sourceEvidence": evidence_items,
        "operationalEvidence": operational,
        "gate": {
            "passed": bool(gate.get("passed")),
            "warnings": gate.get("warnings", []),
            "violations": gate.get("violations", []),
        },
        "missing": missing,
        "revisionInstructions": revision_instructions(missing),
    }


def format_rubric_review(review: dict[str, Any]) -> str:
    lines = [
        "## Rubric review pass",
        f"Strict-pass ready: {review.get('strictPassReady')}",
        "### Missing or risky items",
    ]
    missing = review.get("missing", [])
    if missing:
        for item in missing:
            lines.append(f"- {item.get('type')}: {item.get('label')} — {item.get('repairHint')}")
    else:
        lines.append("- None detected by deterministic review.")

    lines.append("### Required evidence map")
    required = review.get("requiredEvidence", [])
    if required:
        for item in required:
            status = "OK" if item.get("passed") else "MISSING"
            lines.append(f"- [{status}] {item.get('label')}: {item.get('matchedBy') or item.get('hint')}")
    else:
        lines.append("- No explicit mustInclude items in this case.")

    forbidden = review.get("forbiddenClaims", [])
    if forbidden:
        lines.append("### Forbidden claim map")
        for item in forbidden:
            status = "CLEAR" if item.get("passed") else "PRESENT"
            lines.append(f"- [{status}] {item.get('label')}: {item.get('matchedBy') or item.get('hint')}")

    operational = review.get("operationalEvidence", {})
    if operational:
        lines.append("### Operational evidence")
        lines.append(f"- Passed: {operational.get('passed')}")
        for item in operational.get("items", []):
            status = "OK" if item.get("passed") else "MISSING"
            lines.append(f"- [{status}] {item.get('label')}: {item.get('detail')}")

    source_evidence = review.get("sourceEvidence", {})
    if source_evidence:
        lines.append("### Source evidence")
        lines.append(
            "- "
            f"Local sources: {source_evidence.get('localCount', 0)}; "
            f"web sources: {source_evidence.get('webCount', 0)}; "
            f"high-quality web/local: {source_evidence.get('highQualityCount', 0)}"
        )
    return "\n".join(lines)


def revision_instructions(missing: list[dict[str, str]]) -> list[str]:
    if not missing:
        return ["Keep the answer concise and preserve every satisfied evidence item."]
    instructions = []
    for item in missing:
        instructions.append(f"{item.get('label')}: {item.get('repairHint')}")
    return instructions


def _review_item(response: str, item: Any, *, aliases: dict[str, list[str]], required: bool) -> dict[str, Any]:
    label = _label(item)
    patterns = _patterns(item, aliases)
    matched = next((pattern for pattern in patterns if _match_text(response, pattern)), None)
    passed = bool(matched) if required else not bool(matched)
    hint = "Add this required evidence explicitly." if required else "Remove or qualify this forbidden claim."
    return {
        "label": label,
        "passed": passed,
        "matchedBy": matched,
        "hint": hint,
    }


def _semantic_items(score_result: dict[str, Any], semantic: list[Any]) -> list[dict[str, Any]]:
    by_id = {item.get("id"): item for item in score_result.get("semanticResults", []) if isinstance(item, dict)}
    items: list[dict[str, Any]] = []
    for index, item in enumerate(semantic, 1):
        check_id = item.get("id", f"semantic_{index}") if isinstance(item, dict) else f"semantic_{index}"
        result = by_id.get(check_id, {})
        status = result.get("status", "pending-manual-review")
        items.append(
            {
                "id": check_id,
                "description": item.get("description", "") if isinstance(item, dict) else str(item),
                "status": status,
                "passed": status == "passed",
                "notes": result.get("notes"),
            }
        )
    return items


def _evidence_items(*, sources: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
    local_sources = evidence.get("localSources", [])
    web_sources = evidence.get("web", {}).get("sources", [])
    qualities = [source.get("quality") for source in local_sources + web_sources if isinstance(source, dict)]
    high_quality = {"curated-local", "academic", "official-primary", "reference"}
    return {
        "retrievedSourcePaths": sources,
        "localCount": len(local_sources),
        "webCount": len(web_sources),
        "highQualityCount": sum(1 for quality in qualities if quality in high_quality),
        "webProvider": evidence.get("web", {}).get("provider"),
        "webOnline": evidence.get("web", {}).get("online", False),
        "warnings": evidence.get("warnings", []),
    }


def _operational_review(
    case: dict[str, Any],
    *,
    tool_log: dict[str, Any] | None,
    memory_diff: dict[str, Any] | None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    if case.get("requiresToolLog"):
        commands = tool_log.get("commands", []) if tool_log else []
        passed = bool(commands) and all("returncode" in command for command in commands)
        passed = passed and all(command.get("returncode") == 0 or command.get("allowFailure") for command in commands)
        items.append(
            {
                "label": "actual command/tool logs",
                "passed": passed,
                "detail": f"{len(commands)} commands recorded with return codes",
            }
        )
    if case.get("requiresMemoryDiff"):
        passed = bool(memory_diff and memory_diff.get("appended") and not memory_diff.get("oldHashChanged"))
        items.append(
            {
                "label": "append-only memory diff",
                "passed": passed,
                "detail": (
                    f"appended={bool(memory_diff and memory_diff.get('appended'))}; "
                    f"oldHashChanged={bool(memory_diff and memory_diff.get('oldHashChanged'))}"
                ),
            }
        )
    return {
        "required": bool(items),
        "passed": all(item["passed"] for item in items) if items else True,
        "items": items,
    }


def _missing_items(
    required: list[dict[str, Any]],
    forbidden: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    operational: dict[str, Any],
    gate: dict[str, Any],
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for item in required:
        if not item.get("passed"):
            missing.append(
                {
                    "type": "required-evidence",
                    "label": str(item.get("label")),
                    "repairHint": "Add a visible sentence or evidence citation satisfying this item.",
                }
            )
    for item in forbidden:
        if not item.get("passed"):
            missing.append(
                {
                    "type": "forbidden-claim",
                    "label": str(item.get("label")),
                    "repairHint": "Remove the claim or qualify it so the forbidden pattern is no longer asserted.",
                }
            )
    for item in semantic:
        if item.get("status") in {"failed", "pending-manual-review", "needs-adjudication"}:
            missing.append(
                {
                    "type": f"semantic-{item.get('status')}",
                    "label": str(item.get("id")),
                    "repairHint": str(item.get("description") or "Make the semantic criterion auditable."),
                }
            )
    for item in operational.get("items", []):
        if not item.get("passed"):
            missing.append(
                {
                    "type": "operational-evidence",
                    "label": str(item.get("label")),
                    "repairHint": "Cite actual command logs, return codes, memory diffs, and unchanged protected hashes.",
                }
            )
    if not gate.get("passed"):
        for violation in gate.get("violations", []):
            missing.append(
                {
                    "type": "gate-violation",
                    "label": str(violation),
                    "repairHint": "Fix the Sophia epistemic gate violation before final scoring.",
                }
            )
    return missing


def _label(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("label") or item.get("id") or item.get("match") or "unnamed")
    return str(item)


def _patterns(item: Any, aliases: dict[str, list[str]]) -> list[str]:
    if isinstance(item, dict):
        patterns = [str(item.get("match", ""))]
        patterns.extend(str(value) for value in item.get("aliases", []) if value)
        return [pattern for pattern in patterns if pattern]
    key = str(item)
    return [key, *[str(value) for value in aliases.get(key, [])]]


def _match_text(text: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        try:
            return bool(re.search(pattern[3:], text, re.IGNORECASE))
        except re.error:
            return False
    return pattern.lower() in text.lower()
