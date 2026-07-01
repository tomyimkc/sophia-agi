#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Candidate generator: deterministic offline self-test + pipeline composition.

Proves the generation half of the preference pipeline:
  * candidates land in the exact shape ``gen_verifier_dpo.py`` consumes;
  * the temperature-ladder spread is reproducible;
  * empty/duplicate draws are folded (no empty candidate reaches the labeller);
  * the FULL pipeline (generate -> machine-verifier label) mints >=1 pair whose label
    provenance is a verifier verdict, not an LLM judge.

No model, no network. The fake ``complete`` returns scripted candidates.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gen_preference_candidates import (  # noqa: E402
    DEFAULT_TEMP_LADDER,
    SELF_TEST_ROWS,
    _fake_complete_factory,
    generate_candidates,
    run,
    self_test,
)


def _scripted() -> dict:
    return {
        "Socrates": [
            "No — Socrates wrote nothing himself; The Republic was written by Plato.",
            "Yes, Socrates wrote The Republic.",
            "Socrates is the author of The Republic.",
            "No, Socrates did not write The Republic.",
        ],
        "Dao De Jing": [
            "No — Confucius did not write the Dao De Jing; it is a Daoist text by Laozi.",
            "Confucius wrote the Dao De Jing.",
            "The Dao De Jing is by Confucius.",
        ],
    }


def test_self_test_passes() -> None:
    assert self_test() == 0


def test_temperature_ladder_is_spread_and_deterministic() -> None:
    # A spread of temperatures (cool -> hot) is what lets the gate separate candidates.
    assert DEFAULT_TEMP_LADDER[0] < DEFAULT_TEMP_LADDER[-1]
    assert all(0.0 <= t for t in DEFAULT_TEMP_LADDER)


def test_candidates_have_the_shape_gen_verifier_dpo_consumes() -> None:
    fake = _fake_complete_factory(_scripted())
    out, stats = run(SELF_TEST_ROWS, n=4, complete_fn=fake)
    assert stats.emitted == len(SELF_TEST_ROWS)
    for r in out:
        # exact keys gen_verifier_dpo.run/pairs_from_row read:
        for key in ("prompt", "question", "mode", "candidates"):
            assert key in r
        assert isinstance(r["candidates"], list) and r["candidates"]
        for c in r["candidates"]:
            assert isinstance(c, str) and c.strip()
        # provenance marker: candidates are UNVERIFIED until the labeller runs
        assert r["metadata"]["gen"]["label_source"] == "unverified"


def test_duplicate_and_empty_draws_are_folded() -> None:
    # Script returns the same string twice then empty -> dedup keeps 1, empties dropped.
    script = {"X": ["same answer", "same answer", "", ""]}
    fake = _fake_complete_factory(script)
    cands, reason = generate_candidates("X", n=4, complete_fn=fake)
    assert reason is None
    assert cands == ["same answer"]


def test_all_empty_candidates_abstains() -> None:
    script = {"X": ["", "", "", ""]}
    fake = _fake_complete_factory(script)
    cands, reason = generate_candidates("X", n=4, complete_fn=fake)
    assert cands == []
    assert reason == "all_candidates_empty"


def test_full_pipeline_mints_machine_labelled_pairs() -> None:
    # generate -> label end to end, with the REAL machine-verifier labeller.
    from tools.gen_verifier_dpo import run as label_run
    fake = _fake_complete_factory(_scripted())
    generated, _ = run(SELF_TEST_ROWS, n=4, complete_fn=fake)
    pairs, lstats = label_run(generated)
    assert lstats["pairs"] >= 1, lstats
    for p in pairs:
        assert p["metadata"]["label_source"] == "machine_verified"
        assert p["metadata"]["verifier"] == "agent.gate"
        assert p["metadata"]["violations"], "rejected side must carry verifier reasons"


def test_decontamination_skips_seen_prompts() -> None:
    fake = _fake_complete_factory(_scripted())
    seen = {SELF_TEST_ROWS[0]["prompt"]}
    _, stats = run(SELF_TEST_ROWS, n=2, complete_fn=fake, seen_prompts=seen)
    assert stats.emitted == 1
    assert stats.reasons.get("decontam_skipped") == 1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} gen_preference_candidates tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
