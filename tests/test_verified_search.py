# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for served-answer verification — generate → gate → serve|withhold (fail-closed).

Deterministic & offline: generation is a stub (no LLM); retrieval/grounding/verification are
the real, offline components.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import verified_search as vs  # noqa: E402

DNA_BELIEF = {"doNotAttributeTo": ["confucius", "socrates", "plato"]}


def test_dna_violation_is_affirmative_and_negation_aware() -> None:
    assert vs._dna_violation("The Dao De Jing was written by Confucius.", DNA_BELIEF) == "confucius"
    assert vs._dna_violation("Confucius wrote it, allegedly.", DNA_BELIEF) == "confucius"
    assert vs._dna_violation("It is attributed to Confucius.", DNA_BELIEF) == "confucius"
    # denials / contrastive / incidental must NOT fire
    assert vs._dna_violation("It was not written by Confucius.", DNA_BELIEF) is None
    assert vs._dna_violation("Unlike Confucius, Laozi is the author.", DNA_BELIEF) is None
    assert vs._dna_violation("The Dao De Jing is attributed to Laozi.", DNA_BELIEF) is None


class _Chunk:
    def __init__(self, excerpt: str, path: str = "p", title: str = "t") -> None:
        self.excerpt = excerpt
        self.path = path
        self.title = title


def test_verify_answer_passes_for_faithful_clean_answer() -> None:
    chunks = [_Chunk("The Dao De Jing is attributed to Laozi in the Daoist tradition.")]
    v = vs.verify_answer("The Dao De Jing is attributed to Laozi.", question="Who wrote the Dao De Jing?",
                         served_chunks=chunks, belief=DNA_BELIEF)
    assert v["passed"] is True


def test_verify_answer_fails_on_dna_and_on_unfaithful() -> None:
    chunks = [_Chunk("The Dao De Jing is attributed to Laozi.")]
    dna = vs.verify_answer("The Dao De Jing was written by Confucius.", question="q",
                           served_chunks=chunks, belief=DNA_BELIEF)
    assert dna["passed"] is False and dna["dnaViolation"] == "confucius"

    unfaithful = vs.verify_answer("Quantum chromodynamics governs quark interactions today.",
                                  question="q", served_chunks=chunks, belief=None)
    assert unfaithful["passed"] is False and unfaithful["faithful"] is False


def _faithful_dao(_q, _c):
    return "The Dao De Jing is traditionally attributed to Laozi in the Daoist tradition."


def _misattribute(_q, _c):
    return "The Dao De Jing was written by Confucius, the great sage."


def test_verified_answer_serves_faithful_weak_source_hedged() -> None:
    r = vs.verified_answer("Who wrote the Dao De Jing?", _faithful_dao)
    assert r.served is True
    assert r.action == "hedge"  # weak source → hedged even though verified
    assert r.answer.startswith("(low confidence)")


def test_verified_answer_withholds_misattribution() -> None:
    r = vs.verified_answer("Who wrote the Dao De Jing?", _misattribute)
    assert r.served is False
    assert r.action == "withhold"
    assert r.answer is None
    assert r.raw_answer is not None  # raw kept for audit
    assert "confucius" in r.reason


def test_verified_answer_serves_strong_source_committed() -> None:
    def gen(_q, _c):
        return "The atomic bomb was developed during the Manhattan Project in World War II."

    r = vs.verified_answer("Tell me about the atomic bomb", gen)
    assert r.served is True and r.action == "answer"


def test_generation_not_called_when_grounding_abstains() -> None:
    calls = []

    def gen(_q, _c):
        calls.append(1)
        return "should never run"

    # Force grounded search to abstain via impossible thresholds.
    r = vs.verified_answer("Tell me about the atomic bomb", gen, thresholds={"hi": 1.0, "lo": 1.0})
    assert r.action == "abstain"
    assert r.served is False
    assert calls == []  # fail-closed before spending a generation


def test_empty_generation_is_withheld() -> None:
    r = vs.verified_answer("Who wrote the Dao De Jing?", lambda _q, _c: "   ")
    assert r.served is False and r.action == "withhold"


def test_withheld_answer_is_logged_as_gap(tmp_path) -> None:
    from agent.knowledge_gap_log import load_gaps

    ledger = tmp_path / "gaps.jsonl"
    vs.verified_answer("Who wrote the Dao De Jing?", _misattribute, gap_log_path=ledger)
    gaps = load_gaps(ledger)
    assert gaps and gaps[0]["by"] == "verified_search"


def test_to_dict_serializable() -> None:
    import json

    r = vs.verified_answer("Who wrote the Dao De Jing?", _faithful_dao)
    json.dumps(r.to_dict())
