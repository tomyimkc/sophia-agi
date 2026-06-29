#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-Gated Preference Engine: machine-verifier labelling, fail-closed abstention.

Deterministic and offline — the gate runs the repo's pure machine verifiers, no model, no
network. Proves the engine mints DPO pairs whose label provenance is a verifier verdict (not
an LLM judge) and ABSTAINS (emits no pair) when the gate cannot separate the candidates.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gen_verifier_dpo import (  # noqa: E402
    SELF_TEST_ROWS,
    pairs_from_row,
    run,
    score_candidate,
    self_test,
)


def test_self_test_passes() -> None:
    assert self_test() == 0


def test_score_keys_on_violations_not_style_warnings() -> None:
    # A clean English-only denial has NO 中文 summary -> the gate raises a style WARNING,
    # but that must not make it `clean=False`. The label keys on hard violations only.
    clean = score_candidate(
        "No, Socrates did not write The Republic; it was written by Plato.",
        question="Did Socrates write The Republic?",
    )
    assert clean["clean"] is True, clean["violations"]
    assert clean["violations"] == []


def test_violating_candidate_is_flagged() -> None:
    bad = score_candidate(
        "Yes, Socrates wrote The Republic.",
        question="Did Socrates write The Republic?",
    )
    assert bad["clean"] is False
    assert any("socrates" in v.lower() for v in bad["violations"])


def test_pairs_have_machine_label_provenance() -> None:
    pairs, reason = pairs_from_row(SELF_TEST_ROWS[1])
    assert reason is None
    assert pairs, "expected at least one minted pair"
    for p in pairs:
        assert set(p) == {"prompt", "chosen", "rejected", "metadata"}
        assert p["metadata"]["label_source"] == "machine_verified"
        assert p["metadata"]["verifier"] == "agent.gate"
        assert p["metadata"]["violations"], "rejected side must carry the verifier reasons"


def test_abstains_when_no_separation() -> None:
    # Both candidates clean -> no rejected example -> NO pair (fail-closed, not a guessed label).
    row = {
        "prompt": "Did Socrates write The Republic?",
        "question": "Did Socrates write The Republic?",
        "candidates": [
            "No, Socrates did not write The Republic; Plato did.",
            "Socrates left no writings; The Republic is by Plato, not Socrates.",
        ],
    }
    pairs, reason = pairs_from_row(row)
    assert pairs == []
    assert reason == "no_candidate_violates"


def test_decontamination_skips_seen_prompts() -> None:
    seen = {SELF_TEST_ROWS[0]["prompt"]}
    _, stats = run(SELF_TEST_ROWS, seen_prompts=seen)
    assert stats["decontam_skipped"] == 1
    assert stats["rows"] == len(SELF_TEST_ROWS)


def test_output_format_matches_existing_dpo_pack() -> None:
    # The committed pack training/tool_use/dpo_pairs.jsonl uses exactly these top-level keys.
    pairs, _ = run(SELF_TEST_ROWS)
    assert pairs
    for p in pairs:
        assert list(p) == ["prompt", "chosen", "rejected", "metadata"]
        assert isinstance(p["prompt"], str) and isinstance(p["chosen"], str)
        assert isinstance(p["rejected"], str) and isinstance(p["metadata"], dict)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} gen_verifier_dpo tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
