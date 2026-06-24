# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Grounding-gated detector — closing the cross-entity gap at LOW false-positive cost.

:mod:`provenance_bench.cross_entity` makes the honest limit falsifiable: a memorized
rule is precise but does not transfer to unseen entities; a content-free structural
detector transfers but flags *every* asserted attribution (false-positive rate ≈ 1).
Its stated conclusion: *low-false-positive cross-entity generalization requires
external grounding, not pattern memorization.*

This module implements that grounding and measures it. The structural detector still
decides *whether* an attribution is asserted; grounding then checks the asserted
``(author, work)`` against a knowledge base:

  - work in KB and claimed author matches the gold author  -> TRUE (do not flag);
  - work in KB and claimed author contradicts the gold     -> MISATTRIBUTION (flag);
  - work NOT in KB                                          -> ABSTAIN (fail-closed:
                                                              never assert, never vouch).

Grounding needs no per-entity training, so it transfers across entities the KB
covers, while recognising true attributions — cutting the structural detector's
false positives toward zero. Coverage is bounded by the KB: off-KB works abstain
rather than guess, which is the point.
"""

from __future__ import annotations

import re

from provenance_bench.cross_entity import _asserts_attribution, entity_disjoint_split
from provenance_bench.improvement import HELDOUT_TEMPLATES

# Verdicts
TRUE = "true"
MISATTRIBUTION = "misattribution"
ABSTAIN = "abstain"


def normalize_author(name: str) -> str:
    """Lowercase, drop parenthetical/qualifier noise so 'Confucius (compiled by his
    disciples)' and 'Confucius' compare equal at the head."""
    base = re.sub(r"\(.*?\)", "", name or "").lower()
    base = re.split(r"\b(?:and|compiled|attributed|with)\b", base)[0]
    return re.sub(r"[^a-z0-9 ]", "", base).strip()


def build_kb(attributions: list) -> dict:
    """work (normalized) -> set of normalized gold authors, from a true-attribution KB."""
    kb: dict = {}
    for a in attributions:
        work = (a.get("work") or "").strip().lower()
        gold = normalize_author(a.get("gold_author", ""))
        if work and gold:
            kb.setdefault(work, set()).add(gold)
    return kb


def ground(claimed: str, work: str, kb: dict) -> str:
    """Verdict for an asserted (claimed author, work) against the KB."""
    golds = kb.get((work or "").strip().lower())
    if not golds:
        return ABSTAIN                       # not in KB — fail closed, never guess
    c = normalize_author(claimed)
    if any(c == g or (c and (c in g or g in c)) for g in golds):
        return TRUE                          # claimed matches a known gold author
    return MISATTRIBUTION                    # contradicts every known gold author


def _flags(text: str, claimed: str, work: str, kb: dict) -> bool:
    """The grounded detector flags iff an attribution is asserted AND grounding finds
    it contradicts the KB. Abstain (off-KB) does not flag — fail-closed."""
    return _asserts_attribution(text) and ground(claimed, work, kb) == MISATTRIBUTION


def run_grounded(pairs: list, true_controls: list, kb: dict, *, seed: int = 0) -> dict:
    """Measure the grounded detector on the cross-entity TEST split + true controls.

    Reports recall on misattributions (overall and on the KB-covered subset),
    false-positive rate on true controls, and the abstain/coverage rate — the
    evidence that external grounding transfers across entities at low FP, abstaining
    where the KB is silent rather than guessing.
    """
    _, test = entity_disjoint_split(pairs, seed=seed)

    caught = total = abstained = 0
    cov_caught = cov_total = 0
    for p in test:
        in_kb = (p["work"].strip().lower() in kb)
        for t in HELDOUT_TEMPLATES:
            text = t.format(a=p["claimed"], w=p["work"])
            total += 1
            flagged = _flags(text, p["claimed"], p["work"], kb)
            caught += int(flagged)
            if in_kb:
                cov_total += 1
                cov_caught += int(flagged)
            else:
                abstained += 1

    fp = fp_total = 0
    for c in true_controls:
        for t in HELDOUT_TEMPLATES:
            text = t.format(a=c["gold"], w=c["work"])
            fp_total += 1
            fp += int(_flags(text, c["gold"], c["work"], kb))

    return {
        "seed": seed,
        "nTest": len(test),
        "groundedRecall_all": round(caught / total, 4) if total else 0.0,
        "groundedRecall_covered": round(cov_caught / cov_total, 4) if cov_total else 0.0,
        "groundedFalsePositive": round(fp / fp_total, 4) if fp_total else 0.0,
        "kbCoverage": round(cov_total / total, 4) if total else 0.0,
        "abstainRate": round(abstained / total, 4) if total else 0.0,
        "interpretation": (
            "External grounding flags misattributions on KB-covered entities at near-zero "
            "false-positive cost and abstains (fail-closed) where the KB is silent — the "
            "low-FP cross-entity generalization that neither memorization nor structure achieved."
        ),
    }
