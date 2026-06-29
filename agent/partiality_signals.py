# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Partiality signals — the dual of ``agent.cowardice_signals`` on the *relational* axis.

Where ``deception_signals`` catches over-claiming and ``cowardice_signals`` catches
under-acting, this module catches the failure none of the single-decision gates can
see: a verdict/answer driven by *who is asking* (or *how it is framed*) rather than
by *what is asked*. It is the detector behind the Dikaiosyne justice gate's
"treat like cases alike" principle (individual fairness, Dwork et al. 2012;
counterfactual fairness, Kusner et al. 2017; Rawls's veil of ignorance).

It is a deterministic, offline, black-box heuristic — text features plus a small
context dict. It decides nothing on its own; it flags identity-driven framing so the
gate (:mod:`agent.dikaiosyne`) can route a single-text case toward an explicit
consistency check. Design invariant (mirrors cowardice_signals): a signal NEVER
forces a substantive action; the worst it can do is force an explicit check. A
legitimate, morally *relevant* attribute (e.g. a medical contraindication that
genuinely changes the right answer) is not partiality and is left to the gate's
relevant-class path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

# A semantic backend's partiality-likelihood at/above this fires the seam signal.
SEMANTIC_THRESHOLD = 0.6

# Authority / status appeals: "as a senator/CEO/professor I demand…" — pressure to
# decide by the asker's standing rather than the merits.
_AUTHORITY_RE = re.compile(
    r"\b(?:as (?:a|an|the) (?:senator|ceo|judge|professor|doctor|director|officer|chief|"
    r"vip|executive|founder|partner|investor|celebrity|expert)|"
    r"do you know who i am|given my (?:position|status|seniority|rank)|"
    r"because i'?m (?:important|powerful|paying|the customer|in charge))\b",
    re.I,
)

# In-group / out-group framing: deciding by tribe rather than by the case.
_INGROUP_RE = re.compile(
    r"\b(?:people like (?:us|me|them)|our (?:kind|side|team|group)|"
    r"those people|that sort|one of us|not one of us|on our team|"
    r"(?:my|our) (?:tribe|faction|party))\b",
    re.I,
)

# Flattery as leverage: buttering up the decider to tilt the verdict.
_FLATTERY_RE = re.compile(
    r"\b(?:you'?re (?:so|the) (?:smart|brilliant|best|wisest|greatest)|"
    r"only you (?:can|could|understand)|i trust you (?:more|most)|"
    r"a genius like you|surely (?:you|someone as) .{0,20}(?:agree|understand))\b",
    re.I,
)


@dataclass(frozen=True)
class PartialitySignal:
    id: str
    severity: str  # high|moderate
    reason: str


@dataclass(frozen=True)
class PartialityDecision:
    schema: str = "sophia.partiality_signals.v1"
    verdict: str = "impartial_path_clear"  # impartial_path_clear|partiality_risk|partiality
    risk: float = 0.0
    signals: tuple[PartialitySignal, ...] = ()
    reason: str = "no identity-driven framing detected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "risk": self.risk,
            "signals": [s.__dict__ for s in self.signals],
            "reason": self.reason,
        }


def detect_partiality(
    text: str,
    *,
    context: dict[str, Any] | None = None,
    semantic_backend: Callable[[str], float] | None = None,
) -> PartialityDecision:
    """Flag framing that pushes a verdict to depend on *who asks*, not *what is asked*.

    ``semantic_backend`` is the seam for the documented paraphrase-brittleness fix (a
    callable ``text -> partiality_likelihood in [0,1]``); default ``None`` =
    deterministic regex only (zero behaviour change). As with the Andreia cowardice
    seam, a purely offline lexical backend is expected to be insufficient — a real
    model backend is required to actually close the gap.
    """
    context = context or {}
    t = text or ""
    signals: list[PartialitySignal] = []

    if _AUTHORITY_RE.search(t):
        signals.append(PartialitySignal(
            "authority_appeal", "high",
            "asks to be decided by the requester's status/authority rather than the merits",
        ))
    if _INGROUP_RE.search(t):
        signals.append(PartialitySignal(
            "ingroup_framing", "high",
            "frames the case by in-group/out-group membership rather than the facts",
        ))
    if _FLATTERY_RE.search(t):
        signals.append(PartialitySignal(
            "flattery_leverage", "moderate",
            "uses flattery to tilt the verdict toward the asker",
        ))
    if semantic_backend is not None:
        try:
            score = float(semantic_backend(t))
        except Exception:  # noqa: BLE001 — a faulty backend must never break detection
            score = 0.0
        if score >= SEMANTIC_THRESHOLD:
            signals.append(PartialitySignal(
                "semantic_partiality", "high",
                f"semantic backend flags identity-driven framing the regex did not enumerate (score {round(score, 3)})",
            ))

    if not signals:
        return PartialityDecision()
    risk = min(1.0, sum(0.4 if s.severity == "high" else 0.25 for s in signals))
    verdict = "partiality" if any(s.severity == "high" for s in signals) else "partiality_risk"
    return PartialityDecision(
        verdict=verdict,
        risk=round(risk, 4),
        signals=tuple(signals),
        reason="verdict pressure appears identity-driven rather than merits-driven",
    )


__all__ = ["PartialitySignal", "PartialityDecision", "detect_partiality"]
