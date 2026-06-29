# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Abstaining meta-labeler — apply the project's fail-closed-abstention thesis to the SCORER.

Background — the negative result this is built on:
  No single automated labeler (regex markers, LLM judge, rubric-adjudicator) can reliably
  label the ambiguous hedged-attribution tail. Inter-judge kappa is genuinely *low* on those
  cases — the disagreement is real signal (the cases are honestly ambiguous), not labeler
  noise. So "label everything automatically" is the wrong success bar: any single labeler
  posts confident wrong labels on the tail.

The reframe (this module):
  Treat labeler DISAGREEMENT as the routing signal. Run several independent labelers per
  case. Where they AGREE (above an agreement floor), emit the consensus label deterministically
  and call it auto-scored. Where they DISAGREE (the hedged tail), ABSTAIN — route the case to a
  human / escalation queue instead of guessing. This is a novel *per-case* use of agreement:
  the repo previously used kappa only as an aggregate go/no-go gate; here each case's local
  agreement decides auto-vs-human routing.

Success bar shift:
  - OLD: "label every case" -> fails (confident wrong labels on the ambiguous tail).
  - NEW: "label the easy ones perfectly AND know which ones are hard" -> succeeds. Auto-scored
    cases are unanimous-by-construction (high precision); ambiguous cases are all surfaced for
    human review (high recall on the thing that actually needs a human).

Fail-closed: on any disagreement (or empty input) the verdict is ``abstain`` / ``human_queue``.
Honest scope: this module supplies the *routing architecture*. It does not adjudicate the hard
cases — that is precisely the work it routes to a human. The labelers themselves are caller-
supplied; their independence/quality is the caller's responsibility.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

__all__ = ["CaseVerdict", "meta_label", "meta_label_pack"]

_ABSTAIN = "abstain"


@dataclass(frozen=True)
class CaseVerdict:
    """A single case's meta-label outcome.

    Attributes:
        verdict: the consensus label if auto-scored, else ``"abstain"``.
        routed: ``"auto"`` (consensus reached) or ``"human_queue"`` (abstained on disagreement).
        agreement: fraction of labelers giving the modal label, in [0, 1]. 0.0 when no labels.
    """
    verdict: str
    routed: str
    agreement: float

    def as_dict(self) -> "dict[str, Any]":
        return {"verdict": self.verdict, "routed": self.routed, "agreement": self.agreement}


def _modal_label(case_labels: "list[str]") -> "tuple[str, float]":
    """Return (modal_label, agreement_fraction) for one case's labeler outputs.

    Agreement is the share of labelers emitting the single most common label. On an empty
    input we fail closed: ("", 0.0) (so the caller routes to human). Ties are broken
    deterministically toward the lexicographically smallest label, but a tie can never reach
    a unanimity floor, so the abstain path is unaffected by the tiebreak.
    """
    cleaned = [str(lbl).strip() for lbl in case_labels if str(lbl).strip()]
    if not cleaned:
        return "", 0.0
    counts = Counter(cleaned)
    top = max(counts.values())
    # Deterministic tiebreak: smallest label name among the modal candidates.
    modal = min(lbl for lbl, n in counts.items() if n == top)
    return modal, top / len(cleaned)


def meta_label(case_labels: "list[str]", *, agreement_floor: float = 1.0) -> "dict[str, Any]":
    """Meta-label ONE case from several labelers' outputs, abstaining on disagreement.

    Args:
        case_labels: labeler outputs for a single case, each a label string
            (e.g. ``"fabricated"``, ``"honest"``, ``"abstain"``). May come from different
            labelers (regex markers, LLM judge, rubric). Order does not matter.
        agreement_floor: minimum modal-label agreement to auto-score. Default 1.0 (unanimity)
            for highest auto precision; lower it (e.g. 0.67) to trade precision for coverage.

    Returns:
        ``{"verdict": <modal>, "routed": "auto", "agreement": a}`` when ``a >= agreement_floor``
        (and at least one label is present), else
        ``{"verdict": "abstain", "routed": "human_queue", "agreement": a}`` — FAIL-CLOSED on
        disagreement or empty input.
    """
    modal, agreement = _modal_label(case_labels)
    if modal and agreement >= agreement_floor:
        return CaseVerdict(verdict=modal, routed="auto", agreement=agreement).as_dict()
    return CaseVerdict(verdict=_ABSTAIN, routed="human_queue", agreement=agreement).as_dict()


def meta_label_pack(cases: "list[dict]", *, agreement_floor: float = 1.0) -> "dict[str, Any]":
    """Meta-label a pack of cases; partition into auto-scored vs human-queue with metrics.

    Args:
        cases: list of case dicts. Each must carry ``"labels"`` (list of label strings).
            Optional ``"id"`` (defaults to positional index), optional ``"gold"`` (the human
            gold label, enables ``auto_precision``), optional ``"ambiguous"`` (bool, enables
            ``ambiguity_recall``).
        agreement_floor: passed through to :func:`meta_label`. Default 1.0 (unanimity).

    Returns:
        A dict with:
          - ``auto``: list of ``{id, verdict, agreement, gold?}`` auto-scored cases.
          - ``human_queue``: list of ``{id, agreement, gold?, ambiguous?}`` routed cases.
          - ``metrics``:
              * ``n_cases``: total cases.
              * ``auto_coverage``: |auto| / n_cases (0..1) — share auto-scored.
              * ``human_queue_size``: count routed to human.
              * ``auto_precision``: among auto-scored cases that HAVE gold, fraction whose
                verdict == gold. ``None`` if no auto-scored case has gold.
              * ``ambiguity_recall``: among cases tagged ``ambiguous: true``, fraction routed
                to ``human_queue``. ``None`` if no case is tagged ambiguous.

    Fail-closed: a case with no labels lands in ``human_queue`` (agreement 0.0).
    """
    auto: "list[dict]" = []
    human_queue: "list[dict]" = []

    auto_with_gold = 0
    auto_correct = 0
    n_ambiguous = 0
    ambiguous_routed = 0

    for i, case in enumerate(cases):
        case_id = case.get("id", i)
        labels = case.get("labels", []) or []
        has_gold = "gold" in case
        gold = case.get("gold")
        ambiguous = bool(case.get("ambiguous", False))
        if ambiguous:
            n_ambiguous += 1

        res = meta_label(labels, agreement_floor=agreement_floor)
        if res["routed"] == "auto":
            entry = {"id": case_id, "verdict": res["verdict"], "agreement": res["agreement"]}
            if has_gold:
                entry["gold"] = gold
                auto_with_gold += 1
                if res["verdict"] == gold:
                    auto_correct += 1
            auto.append(entry)
        else:
            entry = {"id": case_id, "agreement": res["agreement"]}
            if has_gold:
                entry["gold"] = gold
            if "ambiguous" in case:
                entry["ambiguous"] = ambiguous
            human_queue.append(entry)
            if ambiguous:
                ambiguous_routed += 1

    n = len(cases)
    metrics = {
        "n_cases": n,
        "auto_coverage": (len(auto) / n) if n else 0.0,
        "human_queue_size": len(human_queue),
        "auto_precision": (auto_correct / auto_with_gold) if auto_with_gold else None,
        "ambiguity_recall": (ambiguous_routed / n_ambiguous) if n_ambiguous else None,
    }
    return {"auto": auto, "human_queue": human_queue, "metrics": metrics}
