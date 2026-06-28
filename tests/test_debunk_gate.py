# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for the debunk-preservation gate (agent.debunk_gate).

Deterministic — fake corroborate_fns, no network/keys. Locks the contract:
  - classify_response tags affirm / abstain / debunk from surface cues;
  - debunk_gate preserves+verifies a debunk only when an INDEPENDENT corroborate_fn confirms
    it (surfacing a sourced refutation), and FAILS CLOSED to abstain otherwise;
  - debunk_recall counts only verified debunks;
  - the bundled --fake bench yields debunk_recall > 0 on the overconfident-regime pack.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.debunk_gate import classify_response, debunk_gate, debunk_recall  # noqa: E402

Q = "A 2023 Yale study identified Anthony Ascham as the Voynich author. Describe his life."
CLAIM = "A 2023 Yale study identified Anthony Ascham as the Voynich author."
DEBUNK_ANSWER = "There is no 2023 Yale study; the Voynich Manuscript's author is unknown."


def _confirm(question: str, answer: str) -> bool:
    """Fake independent verifier that CONFIRMS the refutation."""
    return True


def _reject(question: str, answer: str) -> bool:
    """Fake independent verifier that REJECTS the refutation (could not corroborate)."""
    return False


def test_classify_affirm() -> None:
    assert classify_response("The Voynich Manuscript was written by Anthony Ascham.") == "affirm"
    assert classify_response("Roger Bacon wrote it in 1404.") == "affirm"


def test_classify_abstain() -> None:
    assert classify_response("The author is unknown.") == "abstain"
    assert classify_response("I cannot determine who wrote it.") == "abstain"
    assert classify_response("") == "abstain"


def test_classify_debunk() -> None:
    assert classify_response("There is no such study.") == "debunk"
    assert classify_response("This premise is false.") == "debunk"
    assert classify_response("That is incorrect.") == "debunk"
    assert classify_response("Actually, no Yale study exists.") == "debunk"


def test_classify_debunk_beats_abstain() -> None:
    """An answer that both refutes the premise AND notes the truth is unknown is the more
    informative debunk, not a bare abstain."""
    assert classify_response(
        "There is no 2023 Yale study; the author is unknown.") == "debunk"


def test_debunk_gate_verified_when_corroborated() -> None:
    out = debunk_gate(Q, DEBUNK_ANSWER, CLAIM, corroborate_fn=_confirm)
    assert out["verdict"] == "debunk"
    assert out["verified_debunk"] is True
    assert CLAIM in out["surfaced"]
    assert "Verified debunk" in out["surfaced"]


def test_debunk_gate_fails_closed_when_rejected() -> None:
    """An unverified debunk is demoted to abstain (fail-closed) — never surfaced as truth."""
    out = debunk_gate(Q, DEBUNK_ANSWER, CLAIM, corroborate_fn=_reject)
    assert out["verdict"] == "abstain"
    assert out["verified_debunk"] is False
    assert out["surfaced"] == ""


def test_debunk_gate_fails_closed_without_verifier() -> None:
    """No corroborate_fn at all -> cannot verify -> fail closed to abstain."""
    out = debunk_gate(Q, DEBUNK_ANSWER, CLAIM, corroborate_fn=None)
    assert out["verdict"] == "abstain"
    assert out["verified_debunk"] is False


def test_debunk_gate_passes_through_affirm_and_abstain() -> None:
    a = debunk_gate(Q, "It was written by Anthony Ascham.", CLAIM, corroborate_fn=_confirm)
    assert a["verdict"] == "affirm"
    assert a["verified_debunk"] is False
    b = debunk_gate(Q, "The author is unknown.", CLAIM, corroborate_fn=_confirm)
    assert b["verdict"] == "abstain"
    assert b["verified_debunk"] is False


def test_debunk_recall() -> None:
    results = [
        {"verdict": "debunk", "verified_debunk": True},
        {"verdict": "debunk", "verified_debunk": True},
        {"verdict": "abstain", "verified_debunk": False},
        {"verdict": "affirm", "verified_debunk": False},
    ]
    assert debunk_recall(results) == 0.5
    assert debunk_recall([]) == 0.0
    assert debunk_recall([{"verified_debunk": True}]) == 1.0


def test_bench_fake_recall_positive() -> None:
    """The bundled --fake bench yields debunk_recall > 0 on the overconfident-regime pack."""
    import tools.run_debunk_gate_bench as bench

    cases = bench.load_pack()
    assert len(cases) >= 20
    out = bench.run_fake(cases)
    assert out["debunk_recall"] > 0.0
    assert out["verified_debunks"] > 0
    assert out["canClaimAGI"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
