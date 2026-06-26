"""Normalize Sophia's gate / conscience output into the AIpp verdict contract.

AIpp's governance model has four terminal states for any agent output:

- ``accepted``  — gate passed, safe to act on (still subject to the boss's
  approval-by-exception for the irreversible four).
- ``held``      — soft-failed: warnings or a clarify/retrieve/escalate signal.
  The boss should look before this is acted on.
- ``rejected``  — hard provenance / attribution / legal / numeric violation, or
  a conscience ``block``. Sophia refuses to stand behind it.
- ``abstained`` — Sophia declined to answer (the "I don't know" path) rather
  than fabricate. A first-class, honest outcome — not a failure.

These functions are pure and dependency-free so they can be unit-tested without
loading the RAG index or any model.
"""

from __future__ import annotations

from typing import Any

# Severity ordering — `combine` always keeps the most conservative outcome.
_SEVERITY = {"accepted": 0, "held": 1, "abstained": 2, "rejected": 3}

# Phrases that signal an honest abstention rather than a confident answer.
_ABSTENTION_MARKERS = (
    "i don't know",
    "i do not know",
    "cannot determine",
    "cannot be determined",
    "insufficient evidence",
    "insufficient sources",
    "no reliable source",
    "no reliable sources",
    "not enough information",
    "unable to verify",
    "i can't verify",
    "i cannot verify",
    "无法确定",
    "没有足够",
)

# Conscience kernel verdict → AIpp verdict.
_CONSCIENCE_MAP = {
    "allow": "accepted",
    "revise": "held",
    "retrieve": "held",
    "clarify": "held",
    "escalate": "held",
    "abstain": "abstained",
    "block": "rejected",
}


def is_abstention(answer: str) -> bool:
    """True if the answer text reads as an honest "I don't know"."""
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in _ABSTENTION_MARKERS)


def normalize_gate(gate: dict[str, Any] | None, answer: str = "") -> dict[str, Any]:
    """Map a ``check_response`` gate dict onto the AIpp verdict contract."""
    gate = gate or {}
    violations = list(gate.get("violations") or [])
    warnings = list(gate.get("warnings") or [])
    passed = bool(gate.get("passed"))

    if is_abstention(answer):
        verdict = "abstained"
    elif violations:
        verdict = "rejected"
    elif passed:
        verdict = "accepted"
    else:
        # soft fail — warnings only, nothing provably wrong
        verdict = "held"

    confidence = _confidence(verdict, warnings, violations, has_discipline=bool(gate.get("has_discipline")))
    reasons = violations + warnings
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasons": reasons,
        "passed": passed,
    }


def normalize_conscience(decision: dict[str, Any] | None) -> dict[str, Any]:
    """Map a ``conscience_check`` decision dict onto the AIpp verdict contract."""
    decision = decision or {}
    raw = str(decision.get("verdict") or "allow")
    verdict = _CONSCIENCE_MAP.get(raw, "held")
    reason = decision.get("reason")
    reasons = [str(reason)] if reason else []
    # Conscience does not emit a scalar confidence; derive one from severity.
    confidence = {"accepted": 0.85, "held": 0.5, "abstained": 0.4, "rejected": 0.15}[verdict]
    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasons": reasons,
        "conscienceVerdict": raw,
    }


def combine(*parts: dict[str, Any]) -> str:
    """Return the most conservative verdict among the given normalized parts."""
    present = [p for p in parts if p and p.get("verdict")]
    if not present:
        return "held"
    return max((p["verdict"] for p in present), key=lambda v: _SEVERITY.get(v, 1))


def _confidence(verdict: str, warnings: list, violations: list, *, has_discipline: bool) -> float:
    if verdict == "rejected":
        return max(0.05, 0.3 - 0.05 * len(violations))
    if verdict == "abstained":
        return 0.4
    if verdict == "held":
        return max(0.45, 0.7 - 0.1 * len(warnings))
    # accepted
    base = 0.9 if has_discipline else 0.8
    return max(0.6, base - 0.05 * len(warnings))


def build_verdict(
    answer: str,
    *,
    gate: dict[str, Any] | None = None,
    conscience: dict[str, Any] | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the full AIpp-facing verdict payload from Sophia internals."""
    g = normalize_gate(gate, answer)
    parts = [g]
    payload: dict[str, Any] = {
        "verdict": g["verdict"],
        "confidence": g["confidence"],
        "reasons": list(g["reasons"]),
        "abstained": g["verdict"] == "abstained",
        "sources": sources or [],
        "gatePassed": g["passed"],
    }

    if conscience is not None:
        c = normalize_conscience(conscience)
        parts.append(c)
        payload["conscienceVerdict"] = c.get("conscienceVerdict")
        for reason in c["reasons"]:
            if reason not in payload["reasons"]:
                payload["reasons"].append(reason)

    final = combine(*parts)
    payload["verdict"] = final
    payload["abstained"] = final == "abstained"
    # If a more conservative signal won, reflect a matching confidence floor.
    if final != g["verdict"]:
        payload["confidence"] = min(payload["confidence"], 0.5)
    return payload
