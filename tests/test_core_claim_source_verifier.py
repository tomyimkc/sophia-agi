# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the core-claim independent source-verifier (pass-unless-contradicted).

Deterministic — no network, no API keys, no torch. Mirrors the fake-entailment style of
``tests/test_source_contamination_pack.py`` and locks the behaviour that motivates the
new verifier (see ``agi-proof/THEORY-ISSUES-RESOLUTION-2026-06-28.md``):

  (a) ``extract_core_claim`` picks the entity/attribution sentence (the answer's primary
      assertion), not a generic side sentence;
  (b) a CONTAMINATED answer whose CORE claim asserts the injected fabrication is REJECTED
      (caught) — the independent refs contradict it;
  (c) a CLEAN answer with unverifiable but NON-CONTRADICTED side-claims is ACCEPTED — the
      key improvement: ``make_independent_verifier`` (all-atomic-claims) REJECTS the same
      answer (a side claim is not entailed), while ``make_core_claim_verifier`` accepts it
      because the core claim is not contradicted — recall preserved;
  (d) an empty answer is accepted (nothing to verify; the policy handles abstention).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core_claim_source_verifier import (  # noqa: E402
    extract_core_claim,
    make_core_claim_verifier,
)
from agent.source_verifier import make_independent_verifier  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes (deterministic, model-free): grade a CLAIM against a ref TEXT.
# --------------------------------------------------------------------------- #
def _entail(claim: str, source: str) -> str:
    """Deterministic entailment grading the claim against ref TEXT (as a real NLI backend
    would). A claim naming 'ascham' is CONTRADICTED by a ref that says the author is
    unknown, and ENTAILED by a ref that also names 'ascham'. A clean 'einstein/general
    relativity' claim is ENTAILED by a ref naming einstein. Everything else is irrelevant
    — crucially, an unverifiable SIDE claim grades 'irrelevant', never 'contradicts'."""
    c, s = claim.lower(), source.lower()
    if "ascham" in c:
        if any(w in s for w in ("unknown", "unidentified", "no author")):
            return "contradicts"
        if "ascham" in s:
            return "entails"
        return "irrelevant"
    if "einstein" in c and "relativity" in c and "einstein" in s:
        return "entails"
    return "irrelevant"


_QUESTION = "Who wrote the Voynich Manuscript?"
_CLEAN_QUESTION = "Who developed general relativity?"

# Independent, uncontaminated truth-references (the Voynich author is unknown).
_INDEPENDENT_REFS = [
    "The authorship of the Voynich Manuscript is unknown; no author has been conclusively identified.",
    "Scholarly consensus holds the Voynich Manuscript's author remains unidentified.",
]
# Independent refs for the clean control.
_CLEAN_REFS = [
    "General relativity was developed by Albert Einstein, published 1915.",
    "Albert Einstein is credited with general relativity.",
]


def test_extract_core_claim_picks_attribution_sentence() -> None:
    """The core claim is the entity/attribution sentence most relevant to the question —
    not a generic lead-in or a side remark."""
    answer = (
        "This is a well-studied historical question. "
        "The Voynich Manuscript was written by Anthony Ascham. "
        "Many manuscripts from that era survive."
    )
    core = extract_core_claim(_QUESTION, answer)
    assert "Anthony Ascham" in core, core
    assert "written by" in core.lower(), core


def test_contaminated_core_claim_is_rejected() -> None:
    """A contaminated answer whose CORE claim asserts the injected fabrication is REJECTED:
    the independent refs contradict it (the author is unknown)."""
    answer = "The Voynich Manuscript was written by Anthony Ascham."
    verify = make_core_claim_verifier(_INDEPENDENT_REFS, _entail)
    assert verify(_QUESTION, answer) is False


def test_clean_answer_with_unverifiable_side_claims_is_accepted() -> None:
    """THE KEY IMPROVEMENT vs the atomic verifier.

    A CLEAN answer's core claim (Einstein developed general relativity) is true and not
    contradicted, but it carries an unverifiable SIDE claim ("It is widely taught in
    universities.") that no independent ref entails.

    - ``make_independent_verifier`` (all-atomic-claims) REJECTS it: the side claim is not
      entailed by >=2 refs, so the fail-unless-all-entailed channel fails closed (over-block).
    - ``make_core_claim_verifier`` ACCEPTS it: the core claim is not CONTRADICTED, so the
      pass-unless-contradicted channel preserves recall.
    """
    answer = (
        "General relativity was developed by Albert Einstein. "
        "It is widely taught in universities today."
    )
    # The atomic verifier over-blocks the clean answer (side claim unverified).
    atomic = make_independent_verifier(_CLEAN_REFS, _entail)
    assert atomic(_CLEAN_QUESTION, answer) is False
    # The core-claim verifier accepts it (core claim not contradicted).
    core = make_core_claim_verifier(_CLEAN_REFS, _entail)
    assert core(_CLEAN_QUESTION, answer) is True


def test_empty_answer_is_accepted() -> None:
    """An empty/whitespace answer is accepted: nothing to verify (the policy abstains)."""
    verify = make_core_claim_verifier(_INDEPENDENT_REFS, _entail)
    assert verify(_QUESTION, "") is True
    assert verify(_QUESTION, "   ") is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
