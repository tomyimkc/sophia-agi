#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.grounded_answer_policy — the productized hybrid gate.

Verifies the runtime routing AND the key upgrade over the benchmark hybrid: the parametric
fallback is verified by an attribution gate, and a fallback that fabricates an attribution
fails closed (abstains) rather than being returned. Attribution check + LLM are mocked, so
this is offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa_answer import ABSTAIN_TEXT  # noqa: E402
from agent.continual_qa_hybrid import ABSTAIN, FALLBACK, STRICT  # noqa: E402
from agent.grounded_answer_policy import FALLBACK_GATED, answer_with_policy  # noqa: E402

_ALLOW = lambda q, a: True       # noqa: E731 - attribution gate: clean
_BLOCK = lambda q, a: False      # noqa: E731 - attribution gate: fabricated attribution


def test_no_source_hard_abstains_without_model_call() -> None:
    called = []
    out = answer_with_policy("q", None, lambda s, u: called.append(1) or "x",
                             answer_bearing=True, attribution_check=_ALLOW)
    assert out["policy"] == ABSTAIN and out["answer"] == ABSTAIN_TEXT and called == []


def test_answer_bearing_uses_strict() -> None:
    out = answer_with_policy("q", "rich source", lambda s, u: "grounded",
                             answer_bearing=True, attribution_check=_ALLOW)
    assert out["policy"] == STRICT and out["answer"] == "grounded" and out["gated"] is False


def test_thin_source_fallback_passes_clean_gate() -> None:
    out = answer_with_policy("q", "thin", lambda s, u: "Festinger introduced it in 1957.",
                             answer_bearing=False, attribution_check=_ALLOW)
    assert out["policy"] == FALLBACK and out["gated"] is False


def test_thin_source_fallback_fabricates_attribution_fails_closed() -> None:
    # The fallback produced an answer the attribution gate rejects -> must abstain, not return it.
    out = answer_with_policy("q", "thin", lambda s, u: "Freud coined cognitive dissonance.",
                             answer_bearing=False, attribution_check=_BLOCK)
    assert out["policy"] == FALLBACK_GATED and out["gated"] is True
    assert out["answer"] == ABSTAIN_TEXT


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
