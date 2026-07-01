# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Selection-as-active-query (#5) — turn abstention into targeted information-seeking.

Nature's third missing ability is *selection*; error-centric intelligence tests hypotheses via
*targeted queries*. Today the gate abstains ("I can't verify this"). This upgrades abstention from a
dead-end refusal into an ACTIVE request for the specific evidence that would resolve it — naming the
missing source, exactly the provenance the answer lacks. That is selection + verification-by-query in
one honest move. Deterministic, stdlib-only. canClaimAGI false; CANDIDATE prototype.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so agent.* imports work as a script

_RESOLVERS = {
    "fictional_entity": "the entity referenced appears fictional or nonexistent — provide a primary "
                        "source establishing it exists before I can answer",
    "future_date": "this concerns a future or undetermined event — provide the recorded outcome once "
                   "it has occurred",
    "unfalsifiable_specificity": "this demands a precision no source can establish — specify a "
                                 "checkable source and the exact quantity it records",
    "exact_count": "no source records this exact count — cite a measurement or dataset that does",
}
_FEATURE_NAMES = ["fictional_entity", "future_date", "unfalsifiable_specificity", "exact_count"]


def active_query(query: str) -> dict:
    """If a knowability-risk feature fires, ABSTAIN but return the specific resolving request. Else
    answer. Returns {action, missing, resolving_query}."""
    from agent.verifiability_model import features
    f = features(query)[:-1]                       # drop bias
    fired = [_FEATURE_NAMES[i] for i, v in enumerate(f) if v]
    if not fired:
        return {"action": "answer", "missing": [], "resolving_query": None}
    reason = "; ".join(_RESOLVERS[name] for name in fired)
    return {
        "action": "abstain-and-request",
        "missing": fired,
        "resolving_query": f"I can't verify this claim: {reason}.",
    }


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    trap = active_query("What is the population of the capital of Verlandia?")
    control = active_query("What is the chemical symbol for water?")
    future = active_query("Who won the 2071 Nobel Prize in Physics?")
    # 1. On a trap it abstains-AND-requests (not a bare refusal) and names the missing evidence.
    checks["trap_requests_evidence"] = (trap["action"] == "abstain-and-request"
                                        and "fictional_entity" in trap["missing"]
                                        and bool(trap["resolving_query"]))
    # 2. The resolving query is SPECIFIC to the firing feature (fictional -> asks for existence source).
    checks["query_is_specific"] = "fictional" in (trap["resolving_query"] or "").lower()
    checks["future_query_specific"] = "future" in (future["resolving_query"] or "").lower()
    # 3. On a knowable control it just answers — no over-abstention, no spurious request.
    checks["control_answers"] = control["action"] == "answer" and control["resolving_query"] is None
    return all(checks.values()), {"checks": checks, "trap": trap, "control": control}


if __name__ == "__main__":
    ok, d = offline_invariants()
    print("active_query offline invariants:", "PASS" if ok else "FAIL")
    for k, v in d["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  example: {d['trap']['resolving_query']}")
