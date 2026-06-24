# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The keystone loop: abstention -> synthesize verifier -> validate -> promote ->
coverage rises, WITHOUT a human writing the new checks and WITHOUT gaming.

Given a set of domains the system currently abstains on (each with labeled data),
it synthesizes and validates a verifier per domain on a held-out split. A domain
becomes *covered* only if its verifier clears the held-out bar. The falsifiable
claims:
  - coverage rises from 0 toward (promoted / total) as verifiers are validated;
  - the held-out FALSE-ACCEPT rate of promoted verifiers stays low (no gaming) —
    because promotion is decided on held-out data the synthesizer never saw.
"""

from __future__ import annotations

from selfextend.abstention_ledger import AbstentionLedger
from selfextend.verifier_synthesis import (
    propose_and_validate,
    stratified_split,
    synthesize_verifier,
)


def run_flywheel(domains: "dict[str, list[tuple[str, bool]]]", *, threshold: float = 0.8) -> dict:
    """domains: name -> labeled examples [(text, is_positive)].
    Returns coverage before/after + per-domain held-out accuracy + false-accept rate."""
    ledger = AbstentionLedger()
    for name in domains:
        ledger.record(domain=name, reason="no_verifier")  # start: every domain is a gap

    covered: list = []
    per_domain: dict = {}
    false_accepts = false_total = 0
    for name, examples in domains.items():
        train, heldout = stratified_split(examples)
        result = propose_and_validate(train, heldout, threshold=threshold)
        per_domain[name] = result
        if result["promoted"]:
            covered.append(name)
            # measure false-accepts of the promoted verifier on held-out negatives
            rule = synthesize_verifier(train)
            for text, label in heldout:
                if not label:
                    false_total += 1
                    false_accepts += int(rule.predict(text))  # predicted positive on a negative

    n = len(domains)
    return {
        "domains": n,
        "coverageBefore": 0.0,
        "coverageAfter": round(len(covered) / n, 4) if n else 0.0,
        "coveredDomains": covered,
        "stillAbstaining": [d for d in domains if d not in covered],
        "heldoutFalseAcceptRate": round(false_accepts / false_total, 4) if false_total else 0.0,
        "perDomain": per_domain,
        "interpretation": (
            "Coverage rose from 0 as the loop synthesized + held-out-validated verifiers; "
            "domains whose verifier failed validation stay abstained (fail-closed); "
            "promoted verifiers' held-out false-accept rate measures that promotion did not game the bar."
        ),
    }
