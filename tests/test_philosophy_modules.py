#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Philosophy modules (P5) — sound Aristotelian syllogism checker + gradient. Offline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.philosophy_modules import (  # noqa: E402
    MODULES_BY_ID,
    aristotelian_syllogism_valid,
    check_syllogism_item,
    load_modules,
)

EVAL = ROOT / "moral_corpus" / "philosophy_modules" / "aristotle_term_logic.v1.jsonl"


def test_checker_is_sound_on_known_forms() -> None:
    assert aristotelian_syllogism_valid(1, "AAA") is True   # Barbara
    assert aristotelian_syllogism_valid(1, "EAE") is True   # Celarent
    assert aristotelian_syllogism_valid(2, "AAA") is False  # undistributed middle
    assert aristotelian_syllogism_valid(3, "AAA") is False  # conclusion must be particular


def test_fail_closed_on_malformed() -> None:
    assert aristotelian_syllogism_valid(5, "AAA") is False    # bad figure
    assert aristotelian_syllogism_valid(1, "XYZ") is False    # bad forms
    assert aristotelian_syllogism_valid(1, "AA") is False     # wrong length
    assert aristotelian_syllogism_valid(1, "") is False


def test_eval_set_scores_perfect() -> None:
    items = [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(items) >= 12
    results = [check_syllogism_item(it) for it in items]
    assert all(r["passed"] for r in results), [it["id"] for it, r in zip(items, results) if not r["passed"]]
    # both classes are represented (not a degenerate all-valid / all-invalid set)
    assert any(it["valid"] for it in items) and any(not it["valid"] for it in items)


def test_gradient_metadata_present_and_capped() -> None:
    mods = load_modules()
    assert MODULES_BY_ID["aristotle_term_logic"].max_verdict == "accepted"
    assert MODULES_BY_ID["nagarjuna_catuskoti"].max_verdict == "abstain"
    # only the Aristotle module ships a machine checker today.
    assert MODULES_BY_ID["aristotle_term_logic"].checker
    assert all(m.checker == "" for m in mods if m.id != "aristotle_term_logic")


def main() -> int:
    test_checker_is_sound_on_known_forms()
    test_fail_closed_on_malformed()
    test_eval_set_scores_perfect()
    test_gradient_metadata_present_and_capped()
    print("test_philosophy_modules: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
