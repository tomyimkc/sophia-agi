#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the live provenance-derived confidence feeding the graded router (S2).

Covers: source-quality priors are monotonic (consensus > legendary); a `contradicts` edge
pulls confidence down (dissent); an unknown target yields no signal; end-to-end, a
weakly-sourced (legendary) answer is downgraded by grounded_answer(confidence_from_sources=
True) while a consensus-sourced answer is kept; and the corpus-wide discrimination invariant
(weak sources always downgraded). Offline, deterministic, LLM mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa_hybrid import STRICT  # noqa: E402
from agent.grounded_agent import grounded_answer  # noqa: E402
from agent.grounded_confidence import (  # noqa: E402
    corroboration_evidence_for, grounded_source_confidence,
)
from okf.page import Page  # noqa: E402

_RICH = ("# {id} (论)\n\nThis is a compiled record of conversations and teachings with "
         "substantial answer-bearing prose describing the work and its context.\n")

_ALLOW = lambda q, a: True  # noqa: E731


def _page(pid, conf, *, body=None, **extra):
    meta = {"id": pid, "pageType": "text", "canonicalTitleEn": pid,
            "domain": "philosophy", "authorConfidence": conf}
    meta.update(extra)
    return Page(path=Path(f"{pid}.md"), meta=meta, body=(body or _RICH.format(id=pid)))


def test_priors_are_monotonic() -> None:
    pages = [_page("strongwork", "consensus"), _page("weakwork", "legendary")]
    hi = grounded_source_confidence("strongwork", pages)
    lo = grounded_source_confidence("weakwork", pages)
    assert hi is not None and lo is not None and hi > 0.7 > lo


def test_unknown_target_has_no_signal() -> None:
    assert grounded_source_confidence("nonexistent", [_page("a", "consensus")]) is None
    assert corroboration_evidence_for("nonexistent", [_page("a", "consensus")]) == []


def test_contradicts_edge_lowers_confidence() -> None:
    # Same tier, but the target contradicts its neighbor -> dissent pulls it down.
    plain = [_page("claimx", "attributed")]
    contested = [_page("claimx", "attributed", contradicts=["rivalx"]),
                 _page("rivalx", "attributed")]
    base = grounded_source_confidence("claimx", plain)
    pulled = grounded_source_confidence("claimx", contested)
    assert base is not None and pulled is not None and pulled < base


def test_end_to_end_weak_source_downgraded_strong_kept() -> None:
    weak = [_page("mythtext", "legendary")]
    strong = [_page("solidtext", "consensus")]
    w = grounded_answer("Tell me about mythtext", lambda s, u: "an answer",
                        pages=weak, attribution_check=_ALLOW,
                        graded=True, confidence_from_sources=True)
    s = grounded_answer("Tell me about solidtext", lambda s, u: "an answer",
                        pages=strong, attribution_check=_ALLOW,
                        graded=True, confidence_from_sources=True)
    assert w["graded"]["applied"] and w["graded"]["action"] in ("hedge", "abstain")
    assert w["policy"] != STRICT  # downgraded label
    assert s["graded"]["action"] == "answer" and s["policy"] == STRICT  # kept


def test_explicit_confidence_overrides_source_signal() -> None:
    strong = [_page("solidtext", "consensus")]
    # Force a low explicit confidence; it must win over the (high) source signal -> abstain.
    out = grounded_answer("Tell me about solidtext", lambda s, u: "an answer",
                          pages=strong, attribution_check=_ALLOW,
                          graded=True, confidence_from_sources=True, confidence=0.1)
    assert out["graded"]["action"] == "abstain"


def test_corpus_discrimination_invariant() -> None:
    # The whole-corpus measurement: weak sources are always downgraded; the signal exists.
    from tools.eval_graded_confidence import run

    report = run()
    assert report["weakSources"]["downgradeRate"] == 1.0
    assert report["strongSources"]["keepRate"] >= 0.5
    # monotonic-ish: consensus mean confidence exceeds legendary mean confidence
    tiers = report["perTier"]
    assert tiers["consensus"]["meanConfidence"] > tiers["legendary"]["meanConfidence"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
