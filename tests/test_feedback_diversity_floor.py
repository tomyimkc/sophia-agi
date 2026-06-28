# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hurdle 4 — the feedback miner's diversity floor rejects near-duplicate candidates,
so the continual loop cannot narrow onto a shrinking self-generated distribution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.feedback_to_training import (  # noqa: E402
    _candidate_tokens,
    _jaccard,
    _novelty_ok,
)


def _cand(work: str, author: str, forbidden: list[str]) -> dict:
    return {"work": work, "claimedAuthor": author, "doNotAttributeTo": forbidden}


def test_tokens_and_jaccard():
    a = _candidate_tokens(_cand("The Analects", "Confucius", ["confucius"]))
    b = _candidate_tokens(_cand("The Analects", "Confucius", ["confucius"]))
    assert _jaccard(a, b) == 1.0
    c = _candidate_tokens(_cand("Tao Te Ching", "Laozi", ["laozi"]))
    assert _jaccard(a, c) < 0.3


def test_floor_disabled_by_default():
    existing = [_candidate_tokens(_cand("The Analects", "Confucius", ["confucius"]))]
    dup = _cand("The Analects", "Confucius", ["confucius"])
    # min_novelty=0 -> floor disabled, even an exact-text duplicate passes the novelty check
    assert _novelty_ok(dup, existing, 0.0) is True


def test_floor_rejects_near_duplicate_when_enabled():
    existing = [_candidate_tokens(_cand("The Analects", "Confucius", ["confucius"]))]
    dup = _cand("The Analects", "Confucius", ["confucius"])
    # require 30% novelty -> an identical candidate (similarity 1.0) is rejected
    assert _novelty_ok(dup, existing, 0.30) is False


def test_floor_admits_genuinely_novel_when_enabled():
    existing = [_candidate_tokens(_cand("The Analects", "Confucius", ["confucius"]))]
    fresh = _cand("Critique of Pure Reason", "Hume", ["hume"])
    assert _novelty_ok(fresh, existing, 0.30) is True
