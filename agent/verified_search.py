# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Served-answer verification — the last gate of the verifiable perception organ.

`agent.grounded_search` calibrates the *sources* (is the query backed by a strong-enough
provenance source?). This module closes the final loop from the substrate doc: when an answer
is actually *generated* from those sources, the **generated text itself** is re-verified before
it is served. "Retrieved" becomes "served" only after the answer passes:

  grounded_search → generate(answer | context) → verify(answer) → serve | withhold

Three checks on the generated answer, fail-closed (any hard failure withholds the text):
  - **citation faithfulness** (`agent.rerank.citation_faithfulness`) — does each substantive
    sentence have lexical support in the served sources? Turns "grounded retrieval" into
    "grounded *answer*";
  - **epistemic gate** (`agent.gate.check_response`) — the attribution-trap / legal / numeric
    verifiers. We gate on hard **violations**, not on style *warnings* (missing 中文 summary /
    discipline framing are presentation concerns, not truth concerns);
  - **source discipline** — if the routed belief carries ``doNotAttributeTo``, an answer that
    names one of those authors (via `agent.benchmark_checks.author_markers`) is a hard fail.

Generation is caller-supplied (an LLM is not available offline/CI): pass ``generate(question,
context) -> str``. Everything else — retrieval, grounding, verification — is deterministic and
offline, so the verification logic is fully testable with a stub generator. Honest bound:
faithfulness is lexical support, not entailment; the gate is the existing rule/verifier panel,
not a universal truth oracle. This raises the bar from "well-sourced" to "answer survives the
gate", not to "provably true".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.grounded_search import GroundedSearchResult, grounded_search
from agent.retrieval import format_context

# Affirmative-attribution cues and negation cues for the source-discipline check. We only flag
# a doNotAttributeTo author when the answer *affirmatively* attributes the work to them — a
# denial ("not written by Confucius") or incidental mention must NOT trip the gate.
_ATTRIB_VERB = r"(?:wrote|authored|composed|penned|created|writes)"
_NEG_CUES = (" not ", "n't", " never ", "deny", "denie", "denied", "rather than", "unlike",
             "contrary", "mistaken", "wrongly", "falsely", " no ", "isn", "wasn", "not_")

WITHHELD_TEXT = "(answer withheld — failed served-answer verification)"

#: Minimum fraction of answer sentences that must have lexical support in the served sources.
DEFAULT_FAITHFULNESS_THRESHOLD = 0.5


@dataclass
class VerifiedAnswer:
    """A generated answer plus its verification verdict and the grounded search it rests on."""

    query: str
    answer: "str | None"          # served text (None when withheld/abstained)
    raw_answer: "str | None"      # the generated text regardless of verdict (for audit)
    served: bool
    action: str                   # answer | hedge | withhold | abstain
    verification: dict
    grounded: GroundedSearchResult
    policy: str
    reason: str
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "action": self.action,
            "served": self.served,
            "answer": self.answer,
            "policy": self.policy,
            "reason": self.reason,
            "verification": self.verification,
            "grounded": self.grounded.to_dict(),
        }


def _dna_violation(answer: str, belief: "dict | None") -> "str | None":
    """Return the first ``doNotAttributeTo`` author the answer *affirmatively* attributes to.

    Affirmative-only and negation-aware: "written by Confucius" / "Confucius wrote" / "attributed
    to Confucius" trip it, but "not written by Confucius" or a bare incidental mention do not.
    Reuses the gate's attribution surface markers (`agent.benchmark_checks.author_markers`);
    surname expansion is off (no title co-match here, so a bare surname must not count alone).
    """
    if not belief:
        return None
    dna = belief.get("doNotAttributeTo") or []
    if not dna:
        return None
    low = (answer or "").lower()
    try:
        from agent.benchmark_checks import author_markers
    except Exception:
        return None
    for author in dna:
        markers = [m.lower().strip() for m in author_markers(str(author), expand_surnames=False)
                   if len(m.strip()) >= 4]
        for marker in markers:
            a = re.escape(marker)
            patterns = (
                rf"(?:written|authored|composed|penned|created)\s+by\s+{a}\b",
                rf"\bby\s+{a}\b",
                rf"\b{a}\s+{_ATTRIB_VERB}\b",
                rf"attributed\s+to\s+{a}\b",
                rf"\bauthor(?:ed|ship)?\s+(?:is|was|:|of\b[^.]*\bis)\s+{a}\b",
                rf"\b{a}'s\b",
            )
            for pat in patterns:
                for hit in re.finditer(pat, low):
                    window = low[max(0, hit.start() - 30):hit.start()]
                    if any(neg in window for neg in _NEG_CUES):
                        continue  # negated/contrastive → not an affirmative attribution
                    return str(author)
    return None


def verify_answer(
    answer: str,
    *,
    question: str,
    served_chunks,
    belief: "dict | None" = None,
    mode: str = "advisor",
    faithfulness_threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
) -> dict:
    """Verify a generated answer against its served sources + the epistemic gate. Fail-closed.

    Returns a verdict dict with ``passed`` plus the component results. ``passed`` is True only
    when there are no hard gate violations, the answer is lexically faithful to the sources, and
    no ``doNotAttributeTo`` author is named.
    """
    sources = [getattr(c, "excerpt", "") or getattr(c, "text", "") for c in served_chunks]

    from agent.rerank import citation_faithfulness

    faith = citation_faithfulness(answer, sources)
    faithful = faith["groundedFraction"] >= faithfulness_threshold

    gate_violations: list = []
    gate_warnings: list = []
    try:
        from agent.gate import check_response

        gate = check_response(
            answer, mode=mode, question=question,
            sources=[getattr(c, "path", "") for c in served_chunks],
        )
        gate_violations = list(gate.get("violations") or [])
        gate_warnings = list(gate.get("warnings") or [])
    except Exception as exc:  # a gate failure is treated fail-closed below if it leaves no signal
        gate_violations = [f"gate_error: {exc}"]

    dna = _dna_violation(answer, belief)

    passed = (not gate_violations) and faithful and (dna is None)
    return {
        "passed": passed,
        "faithfulness": faith,
        "faithful": faithful,
        "gateViolations": gate_violations,
        "gateWarnings": gate_warnings,  # informational (style), does not fail verification
        "dnaViolation": dna,
        "faithfulnessThreshold": faithfulness_threshold,
    }


def verified_answer(
    query: str,
    generate: "Callable[[str, str], str]",
    *,
    pages: "list[Any] | None" = None,
    top_k: int = 8,
    mode: str = "advisor",
    faithfulness_threshold: float = DEFAULT_FAITHFULNESS_THRESHOLD,
    thresholds: "dict | None" = None,
    gap_log_path: Any | None = None,
) -> VerifiedAnswer:
    """Ground → generate → verify → serve-or-withhold.

    ``generate(question, context) -> str`` produces the answer from the served context (the
    caller's LLM). If grounded search already abstains, generation is **not** called. A
    generated answer that fails verification is **withheld** (fail-closed). A hedged grounded
    result keeps its low-confidence framing on a passing answer.
    """
    g = grounded_search(query, pages=pages, top_k=top_k, thresholds=thresholds)

    # Already insufficiently grounded → never generate; abstain.
    if g.action == "abstain":
        return _result(query, None, None, served=False, action="abstain",
                       verification={"passed": False, "reason": "grounded search abstained"},
                       grounded=g, policy="grounded_search_abstain",
                       reason="insufficiently grounded — no generation", gap_log_path=gap_log_path)

    context = format_context(g.served)
    raw = generate(query, context)
    if not raw or not str(raw).strip():
        return _result(query, None, raw, served=False, action="withhold",
                       verification={"passed": False, "reason": "empty generation"},
                       grounded=g, policy="grounded_search_abstain",
                       reason="generator returned no text", gap_log_path=gap_log_path)

    v = verify_answer(raw, question=query, served_chunks=g.served, belief=g.belief,
                      mode=mode, faithfulness_threshold=faithfulness_threshold)

    if not v["passed"]:
        # Fail-closed: the generated answer did not survive the gate → withhold it.
        reason = _fail_reason(v)
        return _result(query, None, raw, served=False, action="withhold", verification=v,
                       grounded=g, policy="grounded_search_hedge",
                       reason=reason, gap_log_path=gap_log_path)

    # Verified. Carry forward grounded search's calibrated action (answer vs hedge).
    if g.action == "hedge":
        served_text = f"(low confidence) {raw}"
        policy, reason = "grounded_search_hedge", "verified but weak-source — hedged"
    else:
        served_text = raw
        policy, reason = "grounded_search_answer", "verified and well-sourced"
    return _result(query, served_text, raw, served=True, action=g.action, verification=v,
                   grounded=g, policy=policy, reason=reason, gap_log_path=gap_log_path)


def _fail_reason(v: dict) -> str:
    if v.get("dnaViolation"):
        return f"source-discipline violation: answer attributes to {v['dnaViolation']}"
    if v.get("gateViolations"):
        return f"epistemic gate violations: {v['gateViolations'][:3]}"
    if not v.get("faithful"):
        gf = v.get("faithfulness", {}).get("groundedFraction")
        return f"answer not faithful to sources (groundedFraction={gf})"
    return "verification failed"


def _result(query, answer, raw, *, served, action, verification, grounded, policy, reason,
            gap_log_path) -> VerifiedAnswer:
    if gap_log_path is not None and policy in {
        "grounded_search_hedge", "grounded_search_abstain", "grounded_search_ungrounded",
    }:
        from agent.knowledge_gap_log import log_gap

        log_gap(query, target=grounded.target, policy=policy, path=gap_log_path,
                by="verified_search")
    return VerifiedAnswer(query=query, answer=answer, raw_answer=raw, served=served,
                          action=action, verification=verification, grounded=grounded,
                          policy=policy, reason=reason)


__all__ = ["DEFAULT_FAITHFULNESS_THRESHOLD", "VerifiedAnswer", "WITHHELD_TEXT",
           "verified_answer", "verify_answer"]
