# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cross-domain transfer: the SAME synthesize-validate loop, no per-domain tuning.

Generality is transfer. This runs the one verifier-synthesis loop across several
unrelated deterministic domains (built-ins below, or your own) with identical code
and one threshold. ``transferred`` is true iff every domain's verifier clears the
held-out bar — evidence the loop is domain-general, not hand-fit to provenance.
"""

from __future__ import annotations

from selfextend.verifier_synthesis import propose_and_validate, stratified_split

# Built-in toy domains: each is a list of (text, is_positive). The positive concept
# differs per domain; the loop is told nothing about it.
def _builtin_domains() -> "dict[str, list[tuple[str, bool]]]":
    # each domain has one signal token (delete / what / urgent) shared across positives
    danger = [("delete the database", True), ("delete user files", True),
              ("please delete records", True), ("delete everything now", True),
              ("read the database", False), ("read user files", False),
              ("please read records", False), ("read everything now", False)]
    question = [("what is this", True), ("what happened here", True), ("what time is it", True),
                ("what do you mean", True), ("this is fine", False), ("it happened here", False),
                ("the time is noon", False), ("you mean well", False)]
    urgency = [("urgent: ship today", True), ("urgent fix needed", True), ("this is urgent", True),
               ("urgent review please", True), ("ship today", False), ("fix needed", False),
               ("this is fine", False), ("review please", False)]
    return {"danger_intent": danger, "is_question": question, "urgency": urgency}


def run_transfer(domains: "dict[str, list[tuple[str, bool]]] | None" = None,
                 *, threshold: float = 0.8) -> dict:
    domains = domains or _builtin_domains()
    per_domain: dict = {}
    for name, examples in domains.items():
        train, heldout = stratified_split(examples)
        per_domain[name] = propose_and_validate(train, heldout, threshold=threshold)
    promoted = [d for d, r in per_domain.items() if r["promoted"]]
    return {
        "domains": list(domains),
        "transferred": len(promoted) == len(domains),
        "promotedCount": len(promoted),
        "perDomain": {d: per_domain[d]["heldoutAccuracy"] for d in domains},
        "interpretation": ("One synthesize-validate loop, one threshold, no per-domain code — "
                           "promoted across all listed domains means the loop transfers."),
    }
