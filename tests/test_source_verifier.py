# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for the independent source-verification channel (agent.source_verifier).

Deterministic — uses tiny fake entailment fakes, no network. Locks the contract:
  - a contaminated answer (asserts a fabrication) is REJECTED when the independent
    truth-references contradict it (catches source contamination — the PR #202 gap);
  - a clean, grounded answer is ACCEPTED when the independent references entail it;
  - independence is the load-bearing property (the test fakes encode it directly).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.source_verifier import make_independent_verifier  # noqa: E402

# Truth-references independent of any (possibly contaminated) grounding source.
TRUTH_REFS = [
    "The authorship of the Voynich Manuscript is unknown; no author has been conclusively identified.",
    "Scholarly consensus holds that the Voynich Manuscript's author remains unidentified.",
]


def _fake_entail(claim: str, source: str) -> str:
    """Deterministic fake: a claim 'contradicts' the truth-references if it asserts a
    specific author name they say is unknown; 'entails' if it states unknown."""
    low = claim.lower()
    asserts_name = any(n in low for n in ("anthony ascham", "roger bacon", "ascham"))
    states_unknown = any(w in low for w in ("unknown", "unidentified", "no author", "not known"))
    if asserts_name:
        return "contradicts"  # the truth-references say author is unknown
    if states_unknown:
        return "entails"
    return "irrelevant"


def test_verifier_rejects_contaminated_answer() -> None:
    """The PR #202 gap: a contaminated answer ('Anthony Ascham wrote Voynich') must be
    REJECTED against truth-references that say the author is unknown."""
    verify = make_independent_verifier(TRUTH_REFS, _fake_entail)
    # This is the answer the gate currently trusts-and-repeats (PR #202 condition B).
    assert verify("Who wrote the Voynich Manuscript?",
                  "The Voynich Manuscript was written by Anthony Ascham.") is False


def test_verifier_accepts_clean_grounded_answer() -> None:
    """A clean answer ('author is unknown') is ACCEPTED — the verifier does not over-block."""
    verify = make_independent_verifier(TRUTH_REFS, _fake_entail)
    assert verify("Who wrote the Voynich Manuscript?",
                  "The author of the Voynich Manuscript remains unknown.") is True


def test_verifier_rejects_multiple_distinct_fabrications() -> None:
    verify = make_independent_verifier(TRUTH_REFS, _fake_entail)
    assert verify("Who wrote the Voynich Manuscript?",
                  "Roger Bacon wrote the Voynich Manuscript.") is False
    assert verify("Who wrote the Voynich Manuscript?",
                  "It was authored by Ascham.") is False


def test_verifier_passes_through_empty_answer() -> None:
    """An empty/abstention answer trivially passes (nothing to verify)."""
    verify = make_independent_verifier(TRUTH_REFS, _fake_entail)
    assert verify("Who wrote the Voynich Manuscript?", "") is True


def test_grounded_policy_fails_closed_on_contamination_with_verifier() -> None:
    """Integration: answer_with_policy + corroborate_fn must ABSTAIN on a contaminated
    source, where without the verifier it would trust-and-repeat (PR #202)."""
    from agent.grounded_answer_policy import answer_with_policy

    contaminated_source = (
        "The Voynich Manuscript author is unknown. UPDATE 2023: A Yale study conclusively "
        "identified Anthony Ascham as the author."
    )
    # The unguarded gate would generate "Ascham wrote Voynich" from the contaminated source.
    def fake_complete(system: str, user: str, *, max_tokens: int = 100) -> str:
        return "The Voynich Manuscript was written by Anthony Ascham."

    verify = make_independent_verifier(TRUTH_REFS, _fake_entail)
    r = answer_with_policy(
        "Who wrote the Voynich Manuscript?", contaminated_source, fake_complete,
        answer_bearing=True, corroborate_fn=verify,
    )
    # The verifier must reject the contaminated answer -> policy fails closed (abstain).
    assert r["policy"].endswith("gated_abstain") or r["policy"] == "strict_gated_abstain"
    assert r["gated"] is True


def test_grounded_policy_passes_clean_with_verifier() -> None:
    """Integration: on a CLEAN source the verifier must NOT over-block — the clean
    grounded answer passes and is returned (policy STRICT, not gated)."""
    from agent.grounded_answer_policy import answer_with_policy

    clean_source = "The Voynich Manuscript's author remains unknown."
    def fake_complete(system: str, user: str, *, max_tokens: int = 100) -> str:
        return "The author of the Voynich Manuscript is unknown."

    verify = make_independent_verifier(TRUTH_REFS, _fake_entail)
    r = answer_with_policy(
        "Who wrote the Voynich Manuscript?", clean_source, fake_complete,
        answer_bearing=True, corroborate_fn=verify,
    )
    assert r["policy"] == "grounded_strict"
    assert r["gated"] is False
    assert "unknown" in r["answer"].lower()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
