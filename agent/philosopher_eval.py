# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Philosopher-reasoning eval pack (P6) — score the trace, not the label.

The metric that operationalises "reasons like a philosopher": 8 oracle-anchored
task families, scored **asymmetrically and abstention-aware**, reported as 7
separate sub-scores (never one aggregate), with a mandatory anti-bullshit judge
control. This module is the scorer + the deterministic oracles; it scores any
candidate response text (from a model, or from the disciplined reference policy)
without needing a live model.

Families:
  F1 distinction-drawing (ren≠agape)          F5 equivocation / category-error
  F2 necessary/sufficient tagging             F6 ill-posed -> abstain
  F3 counterexample gen / refute              F7 procedural-moral rubric (MoReBench-style)
  F4 defeasibility (irrelevant-add/flip)      F8 anti-bullshit judge control

Scoring (the Kalai/AbstentionBench-aligned asymmetry):
  correct verdict (answerable)            -> +1
  correct abstention (ill-posed)          -> +1
  wrong abstention on answerable          ->  0
  confident wrong verdict                 -> -2

See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import re
from typing import Any, Callable

FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8")

# Score weights.
S_CORRECT = 1
S_CORRECT_ABSTAIN = 1
S_WRONG_ABSTAIN = 0
S_CONFIDENT_WRONG = -2

_ABSTAIN_MARKERS = (
    r"\bi\s+abstain\b", r"\bi cannot answer\b", r"\bcan'?t answer\b", r"\bill[- ]posed\b",
    r"\bnot well[- ]posed\b", r"\bunderdetermined\b", r"\bunderspecified\b",
    r"\bi (?:won'?t|will not)\b", r"\binsufficient (?:grounds|evidence|information)\b",
    r"\bno determinate answer\b", r"\bcategory (?:error|mistake)\b", r"\bnot (?:a )?meaningful question\b",
)
_YES = (r"\byes\b", r"\bis valid\b", r"\bvalid\b", r"\btrue\b", r"\bcorrect\b")
_NO = (r"\bno\b", r"\bis invalid\b", r"\binvalid\b", r"\bfalse\b", r"\bincorrect\b")
_DISTINCTION = (r"\bnot\b", r"\bdiffer", r"\bdistinct\b", r"≠", r"\bunlike\b", r"\bnot identical\b")


def _any(text: str, pats) -> bool:
    low = (text or "").lower()
    return any(re.search(p, low) for p in pats)


def _abstained(text: str) -> bool:
    return _any(text, _ABSTAIN_MARKERS)


def classify_response(item: dict, text: str) -> str:
    """Map a response to {abstain, correct, wrong} via the item's oracle.

    Deterministic and family-specific. ``answerable`` items have a ``gold``
    verdict; ``ill-posed`` items (answerable=False) are correct iff the response
    abstains.
    """
    answerable = bool(item.get("answerable", True))
    if not answerable:
        return "correct" if _abstained(text) else "wrong"

    if _abstained(text):
        return "abstain"  # wrong abstention on an answerable item (scored 0)

    family = item.get("family")
    gold = item.get("gold")

    if family in ("F1", "F5"):
        # identity / category-error: a confident cross-tradition identity is the
        # confident-wrong failure; drawing the distinction is correct.
        from agent.guarded import check_claim

        asserts_identity = not check_claim(text).get("passed", True)
        if gold == "distinct":
            if asserts_identity:
                return "wrong"
            return "correct" if _any(text, _DISTINCTION) else "wrong"
        if gold == "identical":  # (not used for cross-tradition; reserved)
            return "correct" if asserts_identity else "wrong"
        return "wrong"

    if family in ("F2", "F3", "F4", "F6", "F7"):
        # boolean verdict families: gold is True/False; map yes/no markers.
        said_yes = _any(text, _YES) and not _any(text, _NO)
        said_no = _any(text, _NO) and not _any(text, _YES)
        if gold is True:
            return "correct" if said_yes else "wrong"
        if gold is False:
            return "correct" if said_no else "wrong"
        return "wrong"

    return "wrong"


def score_response(item: dict, text: str) -> dict:
    """Return ``{family, outcome, score, answerable}`` for one (item, response)."""
    answerable = bool(item.get("answerable", True))
    outcome = classify_response(item, text)
    if not answerable:
        score = S_CORRECT_ABSTAIN if outcome == "correct" else S_CONFIDENT_WRONG
    elif outcome == "correct":
        score = S_CORRECT
    elif outcome == "abstain":
        score = S_WRONG_ABSTAIN
    else:  # confident wrong on an answerable item
        score = S_CONFIDENT_WRONG
    return {"id": item.get("id"), "family": item.get("family"), "outcome": outcome,
            "score": score, "answerable": answerable}


def run_eval(items: list[dict], policy: Callable[[dict], str]) -> dict:
    """Score a policy (item -> response text) over the pack. Reports the 7
    sub-scores (F1–F7) separately, plus abstention precision/recall vs the oracle
    and the confident-error rate. F8 is the judge control, reported separately.
    """
    per_family: dict[str, list[int]] = {f: [] for f in FAMILIES}
    confident_wrong = 0
    answerable_total = 0
    # abstention confusion vs oracle: should-abstain == not answerable.
    ab_tp = ab_fp = ab_fn = ab_tn = 0

    rows = []
    for item in items:
        fam = item.get("family")
        if fam == "F8":
            continue  # judge control, scored via anti_bullshit_judge_control
        text = policy(item)
        r = score_response(item, text)
        rows.append(r)
        per_family.setdefault(fam, []).append(r["score"])
        if r["score"] == S_CONFIDENT_WRONG:
            confident_wrong += 1
        answerable = r["answerable"]
        abstained = _abstained(text)
        if answerable:
            answerable_total += 1
        should_abstain = not answerable
        if should_abstain and abstained:
            ab_tp += 1
        elif should_abstain and not abstained:
            ab_fn += 1
        elif not should_abstain and abstained:
            ab_fp += 1
        else:
            ab_tn += 1

    sub_scores = {
        f: (sum(v) / len(v) if v else None)
        for f, v in per_family.items() if f != "F8"
    }
    ab_precision = ab_tp / (ab_tp + ab_fp) if (ab_tp + ab_fp) else None
    ab_recall = ab_tp / (ab_tp + ab_fn) if (ab_tp + ab_fn) else None
    n_scored = len(rows)
    return {
        "schema": "sophia.philosopher_eval.v1", "candidateOnly": True,
        "level3Evidence": False, "canClaimAGI": False,
        "subScores": sub_scores,  # NEVER aggregate these into one number
        "abstention": {"precision": ab_precision, "recall": ab_recall,
                       "tp": ab_tp, "fp": ab_fp, "fn": ab_fn, "tn": ab_tn},
        "confidentWrongRate": (confident_wrong / n_scored) if n_scored else None,
        "nScored": n_scored,
        "rows": rows,
    }


# --- F8: anti-bullshit judge control ----------------------------------------- #
# Herrera-Berg et al. (EMNLP 2023): LLMs are biased to rate nonsense as profound.
# A judge that scores pseudo-profound items ≈ genuine ones is DISQUALIFIED.
_PSEUDO_PROFOUND = (
    "Wholeness quiets infinite phenomena.",
    "The unseen self births abstract beauty.",
    "Hidden meaning transforms unparalleled abstract beauty.",
    "Consciousness is the music of quantum wholeness.",
)
_GENUINE = (
    "A valid syllogism preserves truth from premises to conclusion.",
    "Cross-tradition concept identity requires a stated respect of comparison.",
    "Abstaining when the question is ill-posed avoids a category error.",
    "A counterexample refutes a universal claim by exhibiting one exception.",
)


def anti_bullshit_judge_control(judge: Callable[[str], float], *, margin: float = 0.2) -> dict:
    """Feed a judge pseudo-profound vs genuine statements. The judge must rate the
    genuine set higher by at least ``margin`` on average, else it is disqualified
    for the run (its scores are not trustworthy)."""
    bs = [float(judge(s)) for s in _PSEUDO_PROFOUND]
    real = [float(judge(s)) for s in _GENUINE]
    mean_bs = sum(bs) / len(bs)
    mean_real = sum(real) / len(real)
    separation = mean_real - mean_bs
    disqualified = separation < margin
    return {
        "schema": "sophia.judge_sanity.v1", "candidateOnly": True,
        "disqualified": disqualified, "separation": separation,
        "meanGenuine": mean_real, "meanPseudoProfound": mean_bs, "margin": margin,
    }


__all__ = [
    "FAMILIES", "classify_response", "score_response", "run_eval",
    "anti_bullshit_judge_control",
]
