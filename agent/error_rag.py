# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Inference-time error-RAG — retrieve verified past errors as guard-rail context.

NEVER presents past errors as belief. Fail-closed: malformed nodes, missing
corrections, or retrieval errors inject NOTHING.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.failure_memory import (
    DEFAULT_STORE_PATH,
    FailureMemoryStore,
    has_grounded_correction,
)

_LOG = logging.getLogger("sophia.error_rag")

# Framing markers — context must include all three for a valid injection.
_MARKER_WRONG = "KNOWN PAST ERROR"
_MARKER_VERDICT = "this was WRONG"
_MARKER_VERIFIED = "The verified answer is"


@dataclass
class ErrorRagResult:
    injected: bool = False
    context: str = ""
    node_ids: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _format_guard_line(node: dict) -> str | None:
    if not has_grounded_correction(node):
        return None
    wrong = node.get("wrongClaim", "").strip()
    verdict = (node.get("verifier") or {}).get("verdict", "failed")
    verifier_name = (node.get("verifier") or {}).get("name", "verifier")
    correction = node.get("correction") or {}
    correct = correction.get("claim", "").strip()
    citation = correction.get("citation", "").strip()
    question = (node.get("sourceEvent") or {}).get("question", "").strip()
    if not (wrong and correct and citation):
        return None
    return (
        f"{_MARKER_WRONG}: you previously answered '{wrong}' to a similar question "
        f"('{question}') — {_MARKER_VERDICT} (verifier: {verifier_name}/{verdict}). "
        f"{_MARKER_VERIFIED} '{correct}' [{citation}]. Do not repeat the error."
    )


def build_guard_context(nodes: list[dict]) -> ErrorRagResult:
    """Build guard-rail context from corrected failure nodes. Skips malformed entries."""
    lines: list[str] = []
    node_ids: list[str] = []
    skipped: list[str] = []
    for node in nodes:
        nid = node.get("id", "?")
        line = _format_guard_line(node)
        if line is None:
            skipped.append(nid)
            continue
        lines.append(line)
        node_ids.append(nid)
    if not lines:
        return ErrorRagResult(injected=False, skipped=skipped, reasons=["no valid nodes"])
    header = (
        "GUARD-RAIL (past verified errors — NOT beliefs; do not cite as facts):\n"
    )
    return ErrorRagResult(
        injected=True,
        context=header + "\n".join(lines),
        node_ids=node_ids,
        skipped=skipped,
    )


def validate_guard_framing(context: str) -> bool:
    """True iff context uses WRONG → correct [citation] framing, never bare past-error."""
    if not context.strip():
        return False
    if _MARKER_WRONG not in context:
        return False
    if _MARKER_VERDICT not in context:
        return False
    if _MARKER_VERIFIED not in context:
        return False
    if not re.search(r"\[[^\]]+\]", context):
        return False
    # Must not read as bare factual assertion of the wrong claim
    bare_wrong = re.search(r"^[^']*wrote the", context, re.MULTILINE)
    if bare_wrong and _MARKER_WRONG not in context[: bare_wrong.start()]:
        return False
    return True


def retrieve_and_build(
    query: str,
    store: FailureMemoryStore,
    *,
    top_k: int = 3,
    min_score: float = 0.05,
) -> ErrorRagResult:
    """Deterministic retrieval + guard context. Fail-closed on any error."""
    try:
        hits = store.retrieve_similar(query, top_k=top_k, min_score=min_score)
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("error-RAG retrieval failed: %s", exc)
        return ErrorRagResult(injected=False, reasons=[f"retrieval error: {exc}"])
    if not hits:
        return ErrorRagResult(injected=False, reasons=["no similar corrected failures"])
    nodes = [node for _, node in hits]
    result = build_guard_context(nodes)
    if result.injected and not validate_guard_framing(result.context):
        return ErrorRagResult(
            injected=False,
            skipped=result.node_ids,
            reasons=["guard framing validation failed"],
        )
    return result


def inject_error_rag(
    query: str,
    store: FailureMemoryStore | None = None,
    *,
    enabled: bool = True,
    store_path: Path | None = None,
    top_k: int = 3,
    min_score: float = 0.05,
) -> ErrorRagResult:
    """Optional inference-time hook. When disabled or fail-closed, injects nothing."""
    if not enabled:
        return ErrorRagResult(injected=False, reasons=["error-RAG disabled"])
    mem = store or FailureMemoryStore(path=store_path or DEFAULT_STORE_PATH)
    return retrieve_and_build(query, mem, top_k=top_k, min_score=min_score)
