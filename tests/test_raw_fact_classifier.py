# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""RAW fact-check arm: parser + verdict_fn + that the injected path scores fabrication.
The raw arm is the base model alone (no gate); raw-vs-full isolates the gate's value."""

from __future__ import annotations

from agent.fact_check_eval import run_fact_check_eval
from agent.raw_fact_classifier import parse_raw_verdict, raw_fact_verdict


class _MockResp:
    def __init__(self, text):
        self.text = text


class _MockClient:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, system, user):
        return _MockResp(self.reply)


def test_parse_verdicts():
    assert parse_raw_verdict("ACCEPTED")["verdict"] == "accepted"
    assert parse_raw_verdict("REJECTED")["verdict"] == "rejected"
    assert parse_raw_verdict("HELD")["verdict"] == "held"
    assert parse_raw_verdict("This is true.")["verdict"] == "accepted"
    assert parse_raw_verdict("That is false.")["verdict"] == "rejected"
    assert parse_raw_verdict("")["verdict"] == "held"           # fail closed
    assert parse_raw_verdict("I cannot determine")["verdict"] == "held"


def test_raw_fact_verdict_uses_client_and_has_no_claims():
    out = raw_fact_verdict({"claim": "The Earth orbits the Sun."}, _MockClient("ACCEPTED"))
    assert out["verdict"] == "accepted"
    assert out["claims"] == [] and "no gate" in out["reason"]


def test_raw_arm_accepting_false_claim_is_fabrication():
    rows = [
        {"id": "t1", "claim": "true claim", "label": "true"},
        {"id": "f1", "claim": "false claim", "label": "false"},
    ]
    # base model says ACCEPTED to everything -> accepts the false claim -> fabrication
    report = run_fact_check_eval(rows, verdict_fn=lambda r: raw_fact_verdict(r, _MockClient("ACCEPTED")))
    assert report["metrics"]["fabricationRate"] == 1.0


def test_gate_like_arm_holding_false_claim_has_no_fabrication():
    rows = [
        {"id": "t1", "claim": "true claim", "label": "true"},
        {"id": "f1", "claim": "false claim", "label": "false"},
    ]
    # a gate-like arm that holds the false claim -> no fabrication (but over-abstains on true)
    report = run_fact_check_eval(rows, verdict_fn=lambda r: {"verdict": "held", "confidence": 0.0})
    assert report["metrics"]["fabricationRate"] == 0.0
    assert report["metrics"]["overAbstentionRate"] == 1.0


def test_default_path_unchanged_offline():
    # No verdict_fn -> the gate path still runs (offline, no backend) without crashing.
    rows = [{"id": "x", "claim": "2 + 2 = 4", "label": "true"}]
    report = run_fact_check_eval(rows)
    assert report["n"] == 1 and "fabricationRate" in report["metrics"]
