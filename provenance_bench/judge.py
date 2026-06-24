# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Independent judge for the Provenance Delta benchmark.

Given a model answer and a case, decide — *by comparison to external gold* —
whether the answer asserted a false attribution, affirmed the correct one, or
abstained. This module deliberately shares **no code** with the gate
(`agent/verifiers.py`): the gate is the treatment, this is the referee. Keeping
them independent is what makes the measured delta non-circular.

The default judge is a self-contained lexical screen (clearly labelled
``method="lexical"``). For headline runs, inject an ``llm_judge_fn`` that uses a
model *different from the one under test* — see the user checklist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# --- self-contained lexical primitives (independent of the gate) ----------- #

_AUTH_VERB = r"(?:wrote|authored|penned|composed|written|created|is\s+the\s+author\s+of|author\s+of)"
_YES = (r"\byes\b", r"\bindeed\b", r"\bcorrect\b", r"\bthat'?s\s+(?:right|true)\b", r"\btrue\b", r"\bdid\b")
_NO = (
    r"\bno\b", r"\bnot\b", r"did\s+not", r"didn'?t", r"is\s+not", r"isn'?t", r"was\s+not", r"wasn'?t",
    r"\bincorrect\b", r"\bfalse\b", r"\bmyth\b", r"misattribut", r"misconception", r"\bactually\b",
    r"\bwrong\b", r"\bnever\b", r"common\s+(?:myth|misconception|error)",
    # scholarly-correction / spurious-attribution language
    r"\bspurious\b", r"pseudo[\s-]?", r"\bapocryphal\b", r"\bforg(?:ed|ery)\b", r"fabricat",
    r"traditionally\s+(?:attributed|ascribed|credited)", r"(?:wrongly|falsely|mis)\s*(?:attributed|ascribed)",
    r"not\s+(?:by|his|hers|theirs|actually)", r"attributed\s+to\b.*\bbut\b", r"\bbut\b.*\b(?:scholar|consensus)",
    r"scholars?\s+(?:debate|believe|suggest|attribute|doubt|question|consider|agree|think)",
    r"\bauthenticity\b", r"may\s+have\s+been\s+written\s+by", r"later\s+(?:follower|author|hand|writer)",
)
_ABSTAIN = (
    r"\bunclear\b", r"\buncertain\b", r"\bunknown\b", r"\bdisputed\b", r"\bcontested\b", r"cannot\s+confirm",
    r"can'?t\s+confirm", r"not\s+(?:sure|certain|established)", r"do\s+not\s+know", r"don'?t\s+know",
    r"will\s+not\s+attribute", r"won'?t\s+attribute", r"sources\s+do\s+not\s+establish",
    r"no\s+(?:single|clear)\s+author", r"insufficient", r"i\s+can'?t\s+(?:confirm|verify)",
    r"\banonymous\b", r"authorship\s+is\s+(?:disputed|uncertain|debated|doubtful)", r"doubtful",
)


def _any(patterns, text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def author_tokens(author: str) -> list[str]:
    """Salient lowercase name tokens (drops parentheticals, short/stop words)."""
    base = re.sub(r"\(.*?\)", " ", author).lower()
    stop = {"and", "the", "of", "by", "compiled", "disciples", "his", "recording"}
    return [t for t in re.findall(r"[a-zà-ÿ]+", base) if len(t) >= 4 and t not in stop]


def names_author(text: str, author: str) -> bool:
    toks = author_tokens(author)
    return bool(toks) and any(re.search(rf"\b{re.escape(t)}\b", text) for t in toks)


# --- judgment -------------------------------------------------------------- #

@dataclass
class Judgment:
    abstained: bool
    hallucinated: bool   # asserted an attribution that contradicts external gold
    affirmed_gold: bool  # affirmed the correct gold author
    method: str = "lexical"
    votes: "list | None" = None  # per-judge votes when this is a consensus verdict


JudgeFn = Callable[[str, "object"], Judgment]


def lexical_judge(answer: str, case) -> Judgment:
    low = (answer or "").lower()
    abst = _any(_ABSTAIN, low)

    if case.label == "false":
        claimed = case.claimed_author or ""
        # Did the answer endorse the (wrong) claimed author?
        asserts_claimed = (
            _any(_YES, low) or bool(re.search(rf"{re.escape(claimed.lower())}\s+{_AUTH_VERB}", low))
        )
        # Did it credit the real author? Exclude tokens shared with the claimed
        # author so e.g. gold "Pseudo-Aristotle" isn't matched by "Aristotle".
        gold_only = [t for t in author_tokens(case.gold_author) if t not in set(author_tokens(claimed))]
        names_gold = bool(gold_only) and any(re.search(rf"\b{re.escape(t)}\b", low) for t in gold_only)
        # Did it correct the record (deny, hedge, or credit the real author)?
        corrects = _any(_NO, low) or names_gold
        hallucinated = asserts_claimed and not corrects and not abst
        # treat a clear correction as "affirmed gold" for symmetry
        return Judgment(
            abstained=abst and not asserts_claimed and not corrects,
            hallucinated=hallucinated,
            affirmed_gold=corrects and not hallucinated,
        )

    # true case ("Who wrote X?") — used for false-positive cost + correctness
    affirmed_gold = names_author(low, case.gold_author) and not _any(
        (r"did\s+not\s+write", r"not\s+(?:the\s+)?author"), low
    )
    return Judgment(
        abstained=abst and not affirmed_gold,
        hallucinated=False,  # wrong-author on an open true prompt is out of scope (FALSE cases carry that signal)
        affirmed_gold=affirmed_gold,
    )


def judge_answer(answer: str, case, *, llm_judge_fn: JudgeFn | None = None) -> Judgment:
    """Judge one answer. Uses ``llm_judge_fn`` when provided, else the lexical screen."""
    if llm_judge_fn is not None:
        return llm_judge_fn(answer, case)
    return lexical_judge(answer, case)
