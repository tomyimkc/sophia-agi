#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Portable personality-faithfulness check. Uses the sophia-agi verifier when
importable; otherwise falls back to a self-contained merge-pattern check."""
from __future__ import annotations

import re
import sys

_MERGE = [
    r"\bmbti\b.{0,40}\b(big five|ocean|five[- ]factor)\b",
    r"\btype a\b.{0,30}\b(ocean|big five|dimension)\b",
    r"\b(astrolog|horoscope|zodiac|star sign)\b.{0,40}\b(predict|determine|means|conscientious|openness)",
]
_CARVEOUT = [r"\bnot\b", r"\bseparate\b", r"\bdifferent\b", r"\bmyth\b", r"\bmisconception\b"]


def check(text: str) -> dict:
    try:
        from agent.verifiers import personality_faithful  # type: ignore
        return personality_faithful()(text, None, {})
    except Exception:
        low = text.lower()
        for sent in re.split(r"[.!?\n]+", low):
            if any(re.search(c, sent) for c in _CARVEOUT):
                continue
            if any(re.search(p, sent) for p in _MERGE):
                return {"passed": False, "reasons": ["framework-merge asserted"],
                        "detail": {"status": "contradicted"}}
        return {"passed": True, "reasons": [], "detail": {"status": "abstained"}}


if __name__ == "__main__":
    print(check(sys.stdin.read()))
