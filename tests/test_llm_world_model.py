# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for the LLM-as-world-model (Cluster D1, the CONTENDER).

Deterministic — no torch, no network, no keys. The completer is a deterministic fake.
Locks the self-consistency uncertainty contract (the signal validated on SimpleQA):
  - a fake completer returning CONSISTENT samples -> a confident, non-abstaining
    prediction;
  - a fake completer returning DISAGREEING samples -> ABSTAIN (self-consistency low);
  - a clear majority (above threshold) -> predict the majority with its agreement as
    confidence;
  - a 50/50 tie -> ABSTAIN (no trustworthy single answer).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.llm_world_model import predict  # noqa: E402


def _scripted(samples: list[str]):
    """A completer that returns the given samples in order (one per call)."""
    it = iter(samples)
    return lambda prompt: next(it)


def test_consistent_samples_are_confident() -> None:
    """All samples agree -> confident, non-abstaining prediction (confidence 1.0)."""
    out = predict("state-x", "action-y", lambda prompt: "next_state_ok", samples=5)
    assert out["abstained"] is False
    assert out["prediction"] == "next_state_ok"
    assert out["confidence"] == 1.0
    assert out["agreement"] == 1.0


def test_disagreeing_samples_abstain() -> None:
    """All five samples differ -> self-consistency near zero -> ABSTAIN."""
    out = predict("state-x", "action-y", _scripted(["a", "b", "c", "d", "e"]), samples=5)
    assert out["abstained"] is True
    assert out["prediction"] is None
    assert out["confidence"] == 0.0


def test_clear_majority_predicts_with_agreement_confidence() -> None:
    """3/5 agree (>= 0.6 threshold) -> predict the majority, confidence = agreement."""
    out = predict("state-x", "action-y",
                  _scripted(["accepted", "accepted", "accepted", "rejected", "held"]),
                  samples=5)
    assert out["abstained"] is False
    assert out["prediction"] == "accepted"
    assert out["confidence"] == 0.6


def test_split_vote_abstains() -> None:
    """A 2/2/1 vote: top label has agreement 0.4 < 0.6 threshold -> ABSTAIN."""
    out = predict("state-x", "action-y",
                  _scripted(["a", "a", "b", "b", "c"]), samples=5)
    assert out["abstained"] is True
    assert out["prediction"] is None


def test_exact_tie_abstains() -> None:
    """A 2/2 tie (even sample count) -> no trustworthy winner -> ABSTAIN."""
    out = predict("state-x", "action-y", _scripted(["a", "a", "b", "b"]), samples=4)
    assert out["abstained"] is True
    assert out["prediction"] is None


def test_normalization_groups_equivalent_completions() -> None:
    """Whitespace/case variants of the same answer count as agreement, not disagreement."""
    out = predict("state-x", "action-y",
                  _scripted(["Accepted", "  accepted ", "ACCEPTED", "accepted", "accepted"]),
                  samples=5)
    assert out["abstained"] is False
    assert out["confidence"] == 1.0
    # The returned prediction is real (un-normalised) text from a winning sample.
    assert out["prediction"].strip().lower() == "accepted"


def test_empty_completions_abstain() -> None:
    """If the completer returns only blanks, there is nothing to vote -> ABSTAIN."""
    out = predict("state-x", "action-y", lambda prompt: "   ", samples=5)
    assert out["abstained"] is True
    assert out["prediction"] is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
