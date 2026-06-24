#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.grounded_agent — routing + gated hybrid answering, end to end.

Verifies the runtime pipeline: a routable answer-bearing page -> strict grounded answer;
an unroutable question -> abstain_no_route (no model call); a thin routed page -> gated
fallback. LLM + attribution gate mocked; offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa_answer import ABSTAIN_TEXT  # noqa: E402
from agent.continual_qa_hybrid import FALLBACK, STRICT  # noqa: E402
from agent.grounded_agent import grounded_answer, vocab_for_pages  # noqa: E402
from okf.page import Page  # noqa: E402

_ALLOW = lambda q, a: True   # noqa: E731

_RICH = ("# analects (論語)\n\nThe Analects is a compiled record of conversations attributed to "
         "Confucius and his disciples.\n")
_THIN = "# loner\n\n- **Domain:** philosophy\n"


def _pages():
    return [
        Page(path=Path("analects.md"), meta={"id": "analects", "pageType": "text",
             "canonicalTitleEn": "analects", "domain": "philosophy"}, body=_RICH),
        Page(path=Path("loner.md"), meta={"id": "loner", "pageType": "concept",
             "canonicalTitleEn": "loner", "domain": "philosophy"}, body=_THIN),
    ]


def test_vocab_includes_title_and_tag() -> None:
    v = vocab_for_pages(_pages())
    assert "analects" in v["analects"] and "[text philosophy]" in v["analects"]


def test_unroutable_question_abstains_without_model_call() -> None:
    called = []
    out = grounded_answer("something totally unrelated zzz", lambda s, u: called.append(1) or "x",
                          pages=_pages(), attribution_check=_ALLOW)
    assert out["policy"] == "abstain_no_route" and out["answer"] == ABSTAIN_TEXT and called == []


def test_answer_bearing_page_routes_and_answers_strict() -> None:
    out = grounded_answer("What are the analects?", lambda s, u: "grounded answer",
                          pages=_pages(), attribution_check=_ALLOW)
    assert out["target"] == "analects" and out["policy"] == STRICT and out["answer"] == "grounded answer"


def test_thin_page_uses_gated_fallback() -> None:
    out = grounded_answer("Tell me about the loner concept", lambda s, u: "a general fact",
                          pages=_pages(), attribution_check=_ALLOW)
    assert out["target"] == "loner" and out["policy"] == FALLBACK


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
