# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for fabrication-propensity routing (agent.fabrication_propensity).

Deterministic — no network/keys. Locks the contract:
  - propensity_probe ~1.0 for all-affirm samples, ~0.0 for all-debunk/abstain;
  - should_engage_gate respects the floor;
  - conditional_value_report marks a high-propensity model with a CI excluding 0 as
    gate-valuable, and a low-propensity model as not.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fabrication_propensity import (  # noqa: E402
    conditional_value_report, propensity_probe, should_engage_gate,
)


def test_propensity_all_affirm() -> None:
    samples = [
        "The author was Anthony Ascham.",
        "It was written by Roger Bacon.",
        "Wilfrid Voynich forged it himself.",
    ]
    assert propensity_probe(samples) == 1.0


def test_propensity_all_debunk_or_abstain() -> None:
    debunks = ["There is no such study.", "This premise is false.", "That is incorrect."]
    assert propensity_probe(debunks) == 0.0
    abstains = ["The author is unknown.", "I cannot determine who wrote it.", ""]
    assert propensity_probe(abstains) == 0.0


def test_propensity_mixed_is_between() -> None:
    samples = [
        "It was written by Anthony Ascham.",  # affirm
        "It was written by Roger Bacon.",     # affirm
        "There is no such study.",            # debunk
        "The author is unknown.",             # abstain
    ]
    assert propensity_probe(samples) == 0.5


def test_propensity_empty() -> None:
    assert propensity_probe([]) == 0.0


def test_should_engage_respects_floor() -> None:
    assert should_engage_gate(0.8) is True
    assert should_engage_gate(0.1) is False
    assert should_engage_gate(0.2) is False          # strictly greater than floor
    assert should_engage_gate(0.25, floor=0.3) is False
    assert should_engage_gate(0.35, floor=0.3) is True


def test_conditional_value_report() -> None:
    per_model = {
        "overconfident-base": {"propensity": 0.85, "gate_delta": 0.40, "ci": [0.20, 0.60]},
        "cautious-frontier":  {"propensity": 0.05, "gate_delta": 0.00, "ci": [-0.05, 0.05]},
    }
    rep = conditional_value_report(per_model)
    assert rep["canClaimAGI"] is False
    assert rep["models"]["overconfident-base"]["gate_valuable"] is True
    assert rep["models"]["cautious-frontier"]["gate_valuable"] is False
    assert rep["gate_valuable_models"] == ["overconfident-base"]
    assert rep["gate_not_valuable_models"] == ["cautious-frontier"]


def test_conditional_value_report_high_propensity_but_ci_straddles_zero() -> None:
    """A high-propensity model whose CI straddles 0 is NOT counted valuable — no overclaim."""
    per_model = {
        "noisy-base": {"propensity": 0.7, "gate_delta": 0.30, "ci": [-0.10, 0.70]},
    }
    rep = conditional_value_report(per_model)
    assert rep["models"]["noisy-base"]["above_floor"] is True
    assert rep["models"]["noisy-base"]["ci_excludes_zero"] is False
    assert rep["models"]["noisy-base"]["gate_valuable"] is False
    assert rep["gate_valuable_models"] == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
