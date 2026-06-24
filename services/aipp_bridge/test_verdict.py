"""Pure unit tests for the AIpp verdict normalization (no model/index load)."""

from __future__ import annotations

from services.aipp_bridge.verdict import (
    build_verdict,
    combine,
    is_abstention,
    normalize_conscience,
    normalize_gate,
)


def test_passed_gate_is_accepted():
    out = normalize_gate({"passed": True, "warnings": [], "violations": []}, "Per Dao De Jing, ...")
    assert out["verdict"] == "accepted"
    assert out["confidence"] >= 0.6


def test_violations_are_rejected():
    gate = {"passed": False, "warnings": [], "violations": ["Merged Laozi with Confucius"]}
    out = normalize_gate(gate, "The Dao De Jing was written by Confucius.")
    assert out["verdict"] == "rejected"
    assert out["confidence"] < 0.4


def test_warnings_only_is_held():
    gate = {"passed": False, "warnings": ["Missing 中文 summary section"], "violations": []}
    out = normalize_gate(gate, "Here is an answer.")
    assert out["verdict"] == "held"


def test_abstention_detected_over_pass():
    gate = {"passed": True, "warnings": [], "violations": []}
    out = normalize_gate(gate, "I don't know — there is insufficient evidence to attribute this.")
    assert out["verdict"] == "abstained"
    assert is_abstention("I cannot verify that claim")


def test_conscience_block_maps_to_rejected():
    out = normalize_conscience({"verdict": "block", "reason": "deontic violation"})
    assert out["verdict"] == "rejected"
    assert out["conscienceVerdict"] == "block"


def test_conscience_escalate_maps_to_held():
    assert normalize_conscience({"verdict": "escalate"})["verdict"] == "held"


def test_combine_takes_most_conservative():
    assert combine({"verdict": "accepted"}, {"verdict": "rejected"}) == "rejected"
    assert combine({"verdict": "accepted"}, {"verdict": "held"}) == "held"
    assert combine({"verdict": "accepted"}, {"verdict": "accepted"}) == "accepted"


def test_build_verdict_merges_gate_and_conscience():
    payload = build_verdict(
        "An answer with sources.",
        gate={"passed": True, "warnings": [], "violations": []},
        conscience={"verdict": "escalate", "reason": "high-risk uncertainty"},
        sources=[{"path": "docs/x.md"}],
    )
    # gate says accepted, conscience says held → held wins (conservative)
    assert payload["verdict"] == "held"
    assert payload["confidence"] <= 0.5
    assert "high-risk uncertainty" in payload["reasons"]
    assert payload["sources"] == [{"path": "docs/x.md"}]


def test_build_verdict_accepted_path():
    payload = build_verdict(
        "Per the cited source, ...",
        gate={"passed": True, "warnings": [], "violations": [], "has_discipline": True},
    )
    assert payload["verdict"] == "accepted"
    assert payload["abstained"] is False
    assert payload["gatePassed"] is True
