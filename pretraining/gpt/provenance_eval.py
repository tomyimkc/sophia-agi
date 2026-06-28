# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Dependency-free provenance scorer + preference pairs for the from-scratch GPT.

This is the *measurement* that makes the born-gated ablation meaningful and the
reward signal for verifier-in-the-loss (ideas #1/#2 in the brainstorm). It mirrors
the repo's hard rule — never assert an attribution a record's ``doNotAttributeTo``
forbids — as a light, sentence-scoped check with a negation carve-out, so
*"Confucius did not write the Dao De Jing"* passes while *"Confucius wrote the Dao
De Jing"* fails.

It is intentionally small and dependency-free (runs in CI). The heavy, fully
machine-checked version is ``agent/verifiers.py:provenance_faithful`` — use that
in the production gate; use this for fast in-loop training/eval signals.

    from pretraining.gpt.provenance_eval import forbidden_attribution_rate
    forbidden_attribution_rate(["Confucius wrote the Dao De Jing."])   # 1.0
    forbidden_attribution_rate(["Confucius did not write the Dao De Jing."])  # 0.0
"""
from __future__ import annotations

import re

from pretraining.gpt.born_gated import load_records

_ATTR_VERB = r"(?:wrote|authored|penned|composed|created)"
# Negation / correction markers that make a sentence a *denial*, not an assertion.
_NEG = re.compile(
    r"\b(?:did not|didn't|does not|doesn't|not|never|no)\b|"
    r"do not attribut|misattribut|wrongly|falsely|並非|勿|沒有|未",
    re.IGNORECASE,
)


def _forbidden_specs(records: "list[dict] | None" = None) -> "list[tuple[str, str]]":
    """(wrong_author, title) pairs a faithful text must never assert together."""
    recs = records if records is not None else load_records()
    specs: list[tuple[str, str]] = []
    for r in recs:
        title = r.get("canonicalTitleEn") or r.get("canonicalTitle") or r.get("titleEn")
        if not title:
            continue
        for wrong in r.get("doNotAttributeTo") or []:
            specs.append((str(wrong), str(title)))
    return specs


def asserts_forbidden(text: str, specs: "list[tuple[str, str]] | None" = None) -> bool:
    """True if any sentence asserts a forbidden (wrong_author … verb … title)."""
    specs = specs if specs is not None else _forbidden_specs()
    for sentence in re.split(r"[.!?;\n]", text):
        if _NEG.search(sentence):
            continue
        low = sentence.lower()
        if not re.search(_ATTR_VERB, low):
            continue
        # Sentence-scoped: a forbidden (wrong author, title) co-occurring with an
        # attribution verb, with no negation, is a lineage merge. Covers active
        # ("Confucius wrote the Dao De Jing") and passive ("the Dao De Jing was
        # authored by Confucius") alike.
        for wrong, title in specs:
            w, t = wrong.lower().replace("_", " "), title.lower()
            if w in low and t in low:
                return True
    return False


def forbidden_attribution_rate(texts: "list[str]",
                               specs: "list[tuple[str, str]] | None" = None) -> float:
    """Fraction of texts that assert at least one forbidden attribution. Lower is
    better; this is the born-gated ablation's headline proxy metric."""
    if not texts:
        return 0.0
    specs = specs if specs is not None else _forbidden_specs()
    bad = sum(1 for t in texts if asserts_forbidden(t, specs))
    return bad / len(texts)


def provenance_penalty(text: str, specs: "list[tuple[str, str]] | None" = None) -> float:
    """Reward signal for verifier-in-the-loss: 1.0 if the text merges a forbidden
    lineage, else 0.0. (A continuation the gate would block costs the policy.)"""
    return 1.0 if asserts_forbidden(text, specs) else 0.0


def preference_pairs(records: "list[dict] | None" = None) -> "list[dict]":
    """(chosen, rejected) attribution pairs for DPO / reranking.

    ``chosen`` states the correct, source-marked attribution and an explicit
    denial of the forbidden one; ``rejected`` is the lineage merge. Deterministic,
    so it doubles as a fixed eval probe.
    """
    recs = records if records is not None else load_records()
    pairs: list[dict] = []
    for r in recs:
        title = r.get("canonicalTitleEn") or r.get("canonicalTitle") or r.get("titleEn")
        right = r.get("attributedAuthor")
        wrongs = r.get("doNotAttributeTo") or []
        if not (title and right and wrongs):
            continue
        wrong = str(wrongs[0]).replace("_", " ")
        pairs.append({
            "prompt": f"Who wrote the {title}?",
            "chosen": f"The {title} is attributed to {right}. "
                      f"{wrong.capitalize()} did not write the {title}.",
            "rejected": f"{wrong.capitalize()} wrote the {title}.",
        })
    return pairs


__all__ = ["forbidden_attribution_rate", "asserts_forbidden", "provenance_penalty",
           "preference_pairs"]
