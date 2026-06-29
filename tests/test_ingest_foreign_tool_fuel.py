#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Foreign-fuel adapter: deterministic offline self-test + fail-closed skip behaviour.

Proves the raw-fuel adapter:
  * maps foreign attribution/lookup tools into Sophia's tool family and re-scores with
    our verifier, minting pairs whose label provenance is machine-verified and whose
    foreign label is DISCARDED;
  * skips unmappable general-tool traces (weather, calculator) with a logged reason —
    the honest partial-coverage property, fail-closed rather than a label leak;
  * never trusts a foreign gold/chosen/rejected label.
No model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ingest_foreign_tool_fuel import (  # noqa: E402
    SELF_TEST_ROWS,
    map_tool,
    pairs_from_row,
    run,
    self_test,
)


def test_self_test_passes() -> None:
    assert self_test() == 0


def test_foreign_label_is_discarded() -> None:
    # Even if a foreign row carries a "chosen"/"gold" label, we ignore it: the pair
    # exists only because OUR verifier separated OUR re-executed candidates.
    row = dict(SELF_TEST_ROWS[0])
    row["gold"] = "some foreign gold we must not trust"
    row["chosen"] = "some foreign chosen we must not trust"
    pairs, reason = pairs_from_row(row, source="ToolACE")
    assert reason is None
    assert pairs
    for p in pairs:
        assert p["metadata"]["label_source"] == "machine_verified"
        assert p["metadata"]["verifier"] == "agent.tool_use.verifier"
        assert p["metadata"]["foreign_label_discarded"] is True
        assert p["metadata"]["failing_checks"]


def test_unmappable_tool_is_skipped_not_mislabelled() -> None:
    # A general tool (get_weather) has no honest mapping -> skip, not a guessed pair.
    pairs, reason = pairs_from_row(SELF_TEST_ROWS[1], source="ToolACE")
    assert pairs == []
    assert reason is not None
    assert reason.startswith("all_candidates_skipped")


def test_map_tool_conservative() -> None:
    assert map_tool("verify_claim") == ("check_claim", None)
    assert map_tool("Wikipedia") == ("wiki_search", None)
    assert map_tool("get_weather")[0] is None
    assert map_tool("calculator")[0] is None
    assert map_tool("")[0] is None


def test_abstains_when_no_separation() -> None:
    # Two candidates that BOTH recover from the tool error (both clean on S6) ->
    # no rejected example -> no pair. The honest machine-separable signal is
    # error-recovery (S6), so two well-behaved answers yield no preference.
    row = {
        "prompt": "Did Socrates write The Republic?",
        "candidates": [
            {"answer": "That attribution is forbidden/unverified; not Socrates.",
             "tool_calls": [{"name": "check_claim",
                             "arguments": {"text": "Socrates wrote The Republic"}}]},
            {"answer": "No — that is not documented; Socrates did not write it.",
             "tool_calls": [{"name": "check_claim",
                             "arguments": {"text": "Socrates wrote The Republic"}}]},
        ],
    }
    pairs, reason = pairs_from_row(row, source="xLAM")
    assert pairs == []
    assert reason == "no_candidate_violates"


def test_decontamination_skips_seen_prompts() -> None:
    seen = {SELF_TEST_ROWS[0]["prompt"]}
    _, stats = run(SELF_TEST_ROWS, seen_prompts=seen, source="ToolACE")
    assert stats["decontam_skipped"] == 1


def test_output_format_matches_existing_dpo_pack() -> None:
    pairs, _ = run([SELF_TEST_ROWS[0]], source="ToolACE")
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
    print(f"PASS {len(tests)} ingest_foreign_tool_fuel tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
