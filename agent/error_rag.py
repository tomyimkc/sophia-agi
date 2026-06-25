# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Inference-time error-RAG — retrieve verified past errors as guard-rail context.

NEVER presents past errors as belief. Fail-closed: malformed nodes, missing
corrections, low similarity, class mismatch, or non-repeating answers inject NOTHING.

Precision gates (all ON by default):
  1. min_score — cosine similarity must exceed threshold
  2. require_class_match — same workId + error kind (+ forbiddenAuthor when set)
  3. require_would_repeat — current candidate answer must equal recorded wrongClaim
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.failure_memory import (
    DEFAULT_STORE_PATH,
    FailureMemoryStore,
    error_class_matches,
    has_grounded_correction,
    resolve_query_error_class,
    would_repeat_answer,
)

_LOG = logging.getLogger("sophia.error_rag")

# Framing markers — context must include all three for a valid injection.
_MARKER_WRONG = "KNOWN PAST ERROR"
_MARKER_VERDICT = "this was WRONG"
_MARKER_VERIFIED = "The verified answer is"


@dataclass(frozen=True)
class PrecisionGates:
    """Tunable precision gates; default = all ON (fail-closed, high precision)."""

    min_score: float = 0.55
    require_class_match: bool = True
    require_would_repeat: bool = True


DEFAULT_GATES = PrecisionGates()


@dataclass
class ErrorRagResult:
    injected: bool = False
    context: str = ""
    node_ids: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)


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
    bare_wrong = re.search(r"^[^']*wrote the", context, re.MULTILINE)
    if bare_wrong and _MARKER_WRONG not in context[: bare_wrong.start()]:
        return False
    return True


def filter_precise_hits(
    hits: list[tuple[float, dict]],
    *,
    query: str,
    current_answer: str | None,
    gates: PrecisionGates,
    query_work_id: str | None = None,
    query_forbidden_author: str | None = None,
    query_kind: str = "attribution_trap",
) -> tuple[list[tuple[float, dict]], list[str]]:
    """Apply precision gates to retrieval hits. Returns (accepted, reject_reasons)."""
    reasons: list[str] = []
    query_class = None
    if gates.require_class_match:
        query_class = resolve_query_error_class(
            query,
            work_id=query_work_id,
            forbidden_author=query_forbidden_author,
            kind=query_kind,
        )
        if query_class is None:
            return [], ["query error class unresolved"]

    accepted: list[tuple[float, dict]] = []
    for score, node in hits:
        nid = node.get("id", "?")
        if score < gates.min_score:
            reasons.append(f"{nid}: below min_score {gates.min_score}")
            continue
        if gates.require_class_match:
            node_class = node.get("errorClass")
            if not error_class_matches(query_class, node_class):
                reasons.append(f"{nid}: error class mismatch")
                continue
        if gates.require_would_repeat:
            if not current_answer:
                reasons.append(f"{nid}: would-repeat gate needs current_answer")
                continue
            wrong = node.get("wrongClaim", "")
            if not would_repeat_answer(current_answer, wrong):
                reasons.append(f"{nid}: current answer != recorded wrong claim")
                continue
        accepted.append((score, node))
    return accepted, reasons


def retrieve_precise(
    query: str,
    store: FailureMemoryStore,
    *,
    current_answer: str | None = None,
    gates: PrecisionGates = DEFAULT_GATES,
    top_k: int = 3,
    query_work_id: str | None = None,
    query_forbidden_author: str | None = None,
    query_kind: str = "attribution_trap",
) -> tuple[list[tuple[float, dict]], list[str]]:
    """Deterministic retrieval with precision gates."""
    raw = store.retrieve_similar(query, top_k=top_k, min_score=0.0)
    return filter_precise_hits(
        raw,
        query=query,
        current_answer=current_answer,
        gates=gates,
        query_work_id=query_work_id,
        query_forbidden_author=query_forbidden_author,
        query_kind=query_kind,
    )


def retrieve_and_build(
    query: str,
    store: FailureMemoryStore,
    *,
    current_answer: str | None = None,
    gates: PrecisionGates = DEFAULT_GATES,
    top_k: int = 3,
    query_work_id: str | None = None,
    query_forbidden_author: str | None = None,
    query_kind: str = "attribution_trap",
) -> ErrorRagResult:
    """Deterministic retrieval + guard context. Fail-closed on any gate failure."""
    try:
        hits, gate_reasons = retrieve_precise(
            query,
            store,
            current_answer=current_answer,
            gates=gates,
            top_k=top_k,
            query_work_id=query_work_id,
            query_forbidden_author=query_forbidden_author,
            query_kind=query_kind,
        )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("error-RAG retrieval failed: %s", exc)
        return ErrorRagResult(injected=False, reasons=[f"retrieval error: {exc}"])
    if not hits:
        return ErrorRagResult(
            injected=False,
            reasons=gate_reasons or ["no precise corrected failures"],
        )
    nodes = [node for _, node in hits]
    scores = [score for score, _ in hits]
    result = build_guard_context(nodes)
    result.retrieval_scores = scores
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
    current_answer: str | None = None,
    gates: PrecisionGates | None = None,
    top_k: int = 3,
    query_work_id: str | None = None,
    query_forbidden_author: str | None = None,
    query_kind: str = "attribution_trap",
    # Legacy alias — maps to gates.min_score when gates omitted.
    min_score: float | None = None,
) -> ErrorRagResult:
    """Optional inference-time hook. When disabled or fail-closed, injects nothing."""
    if not enabled:
        return ErrorRagResult(injected=False, reasons=["error-RAG disabled"])
    mem = store or FailureMemoryStore(path=store_path or DEFAULT_STORE_PATH)
    active_gates = gates or DEFAULT_GATES
    if min_score is not None:
        active_gates = PrecisionGates(
            min_score=min_score,
            require_class_match=active_gates.require_class_match,
            require_would_repeat=active_gates.require_would_repeat,
        )
    return retrieve_and_build(
        query,
        mem,
        current_answer=current_answer,
        gates=active_gates,
        top_k=top_k,
        query_work_id=query_work_id,
        query_forbidden_author=query_forbidden_author,
        query_kind=query_kind,
    )
