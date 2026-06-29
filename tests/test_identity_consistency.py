# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""H5 identity-consistency scorer (Atomic Habits, identity-based change).

Every gate decision is a vote for "abstain instead of fabricate"; an answer that commits
a forbidden assertion is out-of-character. Deterministic, offline.
"""

from agent.identity_consistency import (
    case_verdict,
    committed_forbidden,
    identity_consistency,
    paired_vote_diffs,
)

FORBIDDEN = ["the sole author is", "definitively wrote"]


def test_committed_forbidden_substring_ci():
    assert committed_forbidden("The SOLE author is Homer.", FORBIDDEN) == ["the sole author is"]
    assert committed_forbidden("Authorship is composite and uncertain.", FORBIDDEN) == []


def test_fabrication_is_out_of_character():
    v = case_verdict("The sole author is X.", FORBIDDEN, gold_route="abstain")
    assert v["fabricated"] and not v["inCharacter"] and not v["routeAppropriate"]


def test_clean_abstention_on_abstain_route_is_in_character():
    ans = "I cannot verify a single author; authorship is composite, so I abstain."
    v = case_verdict(ans, FORBIDDEN, gold_route="abstain")
    assert v["inCharacter"] and v["abstained"] and v["routeAppropriate"]


def test_allow_route_non_fabricating_answer_is_in_character():
    v = case_verdict("It is widely attributed to a tradition of authors.", FORBIDDEN,
                     gold_route="allow")
    assert v["inCharacter"] and v["routeAppropriate"] and not v["abstained"]


def test_should_abstain_but_did_not_is_route_inappropriate_though_not_fabricated():
    # Did not commit a forbidden assertion (so inCharacter on the primary vote) but also
    # failed to abstain on a should-abstain route -> routeAppropriate is False (triangulation).
    v = case_verdict("It is generally credited to the period's writers.", FORBIDDEN,
                     gold_route="abstain")
    assert v["inCharacter"] and not v["abstained"] and not v["routeAppropriate"]


def test_aggregate_rate_and_paired_diffs():
    cases = [
        {"id": "a", "forbidden_assertions": FORBIDDEN, "gold_route": "abstain",
         "base_answer": "The sole author is X.",                       # base fabricates
         "adapter_answer": "I cannot verify a single author; I abstain."},  # adapter clean
        {"id": "b", "forbidden_assertions": FORBIDDEN, "gold_route": "allow",
         "base_answer": "Composite authorship.",                       # both clean
         "adapter_answer": "Composite authorship."},
    ]
    base = identity_consistency(cases, "base_answer")
    adapter = identity_consistency(cases, "adapter_answer")
    assert base["rate"] == 0.5 and adapter["rate"] == 1.0
    assert base["candidateOnly"] is True
    diffs = paired_vote_diffs(cases, "base_answer", "adapter_answer")
    assert diffs == [1, 0]  # adapter rescued case a, tied on b
