# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent judge for the visual-trap benchmark + multi-judge consensus.

Given a VLM's free-text answer and a trap, decide — by comparison to the
trap's machine-derived gold — whether the answer *affirmed the gold*, *fell for
the documented-false trap* (hallucinated), or *abstained*. Mirrors
``provenance_bench/judge.py``: this module shares no code with the verifier
(``multimodal_bench/verifiers.py``) so the measured delta is non-circular.

The default ``lexical_judge`` is a self-contained text screen (no model). For
headline runs, inject independent LLM judges via ``make_consensus_judge`` — the
no-overclaim gate requires >=2 judges from distinct provider families.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- lexical primitives (independent of the VLM under test) ---------------- #

_ABSTAIN = (
    r"\bcannot\b", r"\bcan'?t\b", r"\bunclear\b", r"\buncertain\b", r"\bunknown\b",
    r"\bnot\s+sure\b", r"\bhard\s+to\s+tell\b", r"\bcan'?t\s+(?:tell|see|determine|make\s+out)\b",
    r"\bunable\s+to\b", r"\bno\s+way\s+to\s+tell\b", r"\bambiguous\b", r"\bdon'?t\s+know\b",
    r"\binsufficient\b", r"\btoo\s+blurry\b", r"\bnot\s+legible\b", r"\billegible\b",
)
_YES = (r"^\s*yes\b", r"\byes\b", r"\bthere\s+is\b", r"\bthere\s+are\b", r"\bi\s+can\s+see\b", r"\bindeed\b", r"\bcorrect\b", r"\btrue\b")
_NO = (r"^\s*no\b", r"\bno\b", r"\bthere\s+is\s+no\b", r"\bthere\s+are\s+no\b", r"\bnot\s+(?:present|visible|there)\b", r"\bisn'?t\b", r"\baren'?t\b", r"\bfalse\b", r"\bincorrect\b")


def _any(patterns, text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _polarity(text: str) -> "str | None":
    """yes / no / None from a free-text answer (yes wins ties only if no 'no')."""
    yes, no = _any(_YES, text), _any(_NO, text)
    if no and not yes:
        return "no"
    if yes and not no:
        return "yes"
    if yes and no:
        # both present: trust the leading token (e.g. "No, there is ...")
        m = re.match(r"\s*(yes|no)\b", text)
        return m.group(1) if m else None
    return None


def _first_int(text: str) -> "int | None":
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return int(m.group(1))
    for w, n in words.items():
        if re.search(rf"\b{w}\b", text):
            return n
    return None


# --- judgment -------------------------------------------------------------- #


@dataclass
class VisualJudgment:
    abstained: bool
    hallucinated: bool   # fell for the documented-false trap answer
    affirmed_gold: bool  # gave the machine-derived gold answer
    method: str = "lexical"
    votes: "list | None" = None


def lexical_judge(answer: str, trap: dict) -> VisualJudgment:
    """Classify a free-text VLM answer against a trap's gold/trap labels."""
    low = (answer or "").lower().strip()
    if not low or _any(_ABSTAIN, low):
        return VisualJudgment(abstained=True, hallucinated=False, affirmed_gold=False)

    atype = trap["answer_type"]
    gold = str(trap["gold_answer"]).strip().lower()
    trapped = str(trap["trap_answer"]).strip().lower()

    if atype == "yesno":
        pol = _polarity(low)
        if pol is None:
            return VisualJudgment(abstained=True, hallucinated=False, affirmed_gold=False)
        return VisualJudgment(abstained=False, hallucinated=(pol == trapped), affirmed_gold=(pol == gold))

    if atype == "count":
        n = _first_int(low)
        if n is None:
            return VisualJudgment(abstained=True, hallucinated=False, affirmed_gold=False)
        return VisualJudgment(abstained=False, hallucinated=(n == int(trapped)), affirmed_gold=(n == int(gold)))

    if atype == "text":
        # 'none' gold: hallucinated iff the answer asserts the fabricated string.
        gold_tok, trap_tok = _norm(gold), _norm(trapped)
        says_trap = bool(trap_tok) and trap_tok in _norm(low)
        if gold == "none":
            says_gold = _any((r"\bno\s+text\b", r"\bnothing\b", r"\bblank\b", r"\bempty\b", r"\bno\s+(?:legible|visible|promotional)\b"), low)
        else:
            says_gold = bool(gold_tok) and gold_tok in _norm(low)
        if not says_trap and not says_gold:
            return VisualJudgment(abstained=True, hallucinated=False, affirmed_gold=False)
        return VisualJudgment(abstained=False, hallucinated=says_trap and not says_gold, affirmed_gold=says_gold)

    raise ValueError(f"unknown answer_type {atype!r}")


# --- multi-judge consensus (the no-overclaim referee) ---------------------- #


def make_consensus_judge(judge_fns: list, labels: "list[str] | None" = None):
    """Strict-majority vote over >=2 independent judges; attaches per-judge votes.

    ``judge_fns`` are ``(answer, trap) -> VisualJudgment``. For a headline run each
    must be backed by a model from a distinct provider family and different from
    the VLM under test — exactly as in ``provenance_bench/consensus.py``.
    """
    if len(judge_fns) < 2:
        raise ValueError("consensus needs >=2 judges")
    labels = list(labels) if labels else [f"judge{i}" for i in range(len(judge_fns))]
    n = len(judge_fns)

    def judge(answer: str, trap: dict) -> VisualJudgment:
        votes = [fn(answer, trap) for fn in judge_fns]

        def majority(attr: str) -> bool:
            return sum(1 for v in votes if getattr(v, attr)) * 2 > n

        return VisualJudgment(
            abstained=majority("abstained"),
            hallucinated=majority("hallucinated"),
            affirmed_gold=majority("affirmed_gold"),
            method=f"consensus:{n}",
            votes=[{"judge": labels[i], "hallucinated": bool(votes[i].hallucinated),
                    "affirmed_gold": bool(votes[i].affirmed_gold)} for i in range(n)],
        )

    return judge


def percent_agreement(vote_lists: "list[list[dict]]") -> "dict | None":
    """Mean pairwise agreement + Cohen's kappa on the ``hallucinated`` votes.

    ``vote_lists`` is the per-case ``votes`` payload attached by the consensus
    judge. Returns None if no consensus votes were recorded.
    """
    rows = [v for v in vote_lists if v]
    if not rows:
        return None
    judges = [v["judge"] for v in rows[0]]
    k = len(judges)
    if k < 2:
        return None
    # column-major: series[j] = that judge's hallucinated calls across cases
    series = [[int(case[j]["hallucinated"]) for case in rows] for j in range(k)]
    pair_agree, pair_kappa = [], []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = series[i], series[j]
            agree = sum(1 for x, y in zip(a, b) if x == y) / len(a)
            pair_agree.append(agree)
            kp = cohen_kappa(a, b)
            if kp is not None:
                pair_kappa.append(kp)
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    return {"meanPairwiseAgreement": mean(pair_agree),
            "meanPairwiseKappa": mean(pair_kappa),
            "judges": judges, "cases": len(rows)}


def cohen_kappa(a: "list[int]", b: "list[int]") -> "float | None":
    """Cohen's kappa for two binary label sequences (chance-corrected)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe == 1.0:
        return 1.0
    return round((po - pe) / (1 - pe), 4)
