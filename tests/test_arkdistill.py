# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.arkdistill — deterministic, offline tool-output compaction.

Covers the four invariants that make ArkDistill safe for the Sophia core:
  1. determinism      — same input+profile ⇒ byte-identical output, repeatably.
  2. real savings     — ≥60% token reduction on the noisy fixtures (the headline).
  3. fail-open        — bad profile / structured input never loses or expands signal.
  4. claim-blindness  — distill_tool_result never mutates the dict the gate checks.
"""
from __future__ import annotations

import json

import pytest

from agent.arkdistill import (
    PROFILES,
    distill,
    distill_tool_result,
    load_profiles,
    profile_for,
)
from agent.context_manager import estimate_tokens


# --------------------------------------------------------------------------- #
# Fixtures: representative noisy tool outputs
# --------------------------------------------------------------------------- #

NOISY_HTML = (
    "<html><head><style>body{margin:0;padding:0;font:14px}</style>"
    "<script>var x=1;function f(){return fetch('/a').then(r=>r.json())}</script></head>"
    "<body><div class='nav'><span>Home</span><span>About</span></div>"
    "<div class='main'><p>The Dao De Jing is attributed to Laozi.</p>"
    "<p>Scholarly consensus is uncertain about single authorship.</p></div>"
    "<img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA' />"
    "<footer><div><span>&nbsp;</span></div></footer></body></html>"
)

NOISY_LOG = "\n".join(
    [f"2026-06-26T10:00:{i:02d}.123Z \x1b[32mINFO\x1b[0m heartbeat ok" for i in range(60)]
    + ["2026-06-26T10:01:00.000Z \x1b[31mERROR\x1b[0m build failed: missing symbol foo"]
)

NOISY_JSON = json.dumps(
    {
        "status": "error",
        "message": "rate limited",
        "trace_id": "abc-123-def-456",
        "debug": {"headers": {f"x-h{i}": "v" * 40 for i in range(30)}},
        "internal_span": "x" * 2000,
    },
    indent=2,
)


# --------------------------------------------------------------------------- #
# 1. Determinism
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,prof", [
    (NOISY_HTML, "browser_html"),
    (NOISY_LOG, "log_dump"),
    (NOISY_JSON, "json_response"),
])
def test_distill_is_byte_identical_across_runs(text, prof):
    first = distill(text, prof)["compacted"]
    for _ in range(5):
        assert distill(text, prof)["compacted"] == first


def test_distill_is_pure_no_input_mutation():
    before = NOISY_HTML
    distill(NOISY_HTML, "browser_html")
    assert NOISY_HTML == before  # input string object unchanged


# --------------------------------------------------------------------------- #
# 2. Real savings (the headline metric)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,prof", [
    (NOISY_HTML, "browser_html"),
    (NOISY_LOG, "log_dump"),
])
def test_noisy_output_compacts_at_least_60_percent(text, prof):
    d = distill(text, prof)
    assert d["applied"] is True
    assert d["ratio"] >= 0.60, f"{prof}: only saved {d['ratio']:.0%}"
    assert d["saved_tokens"] == d["before_tokens"] - d["after_tokens"]


def test_html_drops_scripts_styles_and_base64_but_keeps_visible_claim():
    out = distill(NOISY_HTML, "browser_html")["compacted"]
    assert "Laozi" in out and "consensus is uncertain" in out  # signal preserved
    assert "<script" not in out and "var x=1" not in out        # noise gone
    assert "base64" not in out and "<style" not in out


def test_log_keeps_the_trailing_error_line():
    out = distill(NOISY_LOG, "log_dump")["compacted"]
    assert "build failed: missing symbol foo" in out  # tail-preserving on purpose
    assert "\x1b[" not in out                          # ANSI stripped


def test_json_projection_keeps_signal_fields_and_drops_debug_noise():
    d = distill(NOISY_JSON, "json_response")
    assert d["applied"] is True
    obj = json.loads(d["compacted"])
    assert obj["status"] == "error" and obj["message"] == "rate limited"
    assert "internal_span" not in obj and "debug" not in obj
    assert d["after_tokens"] < d["before_tokens"]


# --------------------------------------------------------------------------- #
# 3. Fail-open: never lose or expand signal
# --------------------------------------------------------------------------- #

def test_short_text_is_returned_unchanged_not_expanded():
    short = "Laozi is the traditional attribution; authorship is contested."
    d = distill(short, "browser_html")
    assert d["compacted"] == short
    assert d["applied"] is False and d["saved_tokens"] == 0


def test_unknown_profile_name_falls_back_and_never_raises():
    d = distill(NOISY_LOG, "no_such_profile")
    assert isinstance(d["compacted"], str)
    assert d["after_tokens"] <= d["before_tokens"]


def test_compacted_never_has_more_tokens_than_input():
    for text in (NOISY_HTML, NOISY_LOG, NOISY_JSON, "", "x", "短文"):
        d = distill(text)
        assert estimate_tokens(d["compacted"]) <= estimate_tokens(text) or not d["applied"]


def test_load_profiles_falls_back_when_file_missing(tmp_path):
    profs = load_profiles(tmp_path / "does-not-exist.json")
    assert "generic" in profs  # baked-in fallback, no raise


# --------------------------------------------------------------------------- #
# 4. Claim-blindness: the gate-checked dict is never mutated
# --------------------------------------------------------------------------- #

def test_distill_tool_result_does_not_mutate_original():
    original = {"tool": "browser_open", "ok": True, "output": NOISY_HTML}
    snapshot = dict(original)
    clone = distill_tool_result(original)
    assert original == snapshot                       # source untouched
    assert clone is not original
    assert clone["ok"] is True                          # ok preserved verbatim
    assert clone["arkdistill"]["saved_tokens"] > 0
    assert len(clone["output"]) < len(original["output"])


def test_distill_tool_result_preserves_ok_false_and_all_fields():
    original = {"tool": "run_tests", "ok": False, "error": "timeout", "output": NOISY_LOG}
    clone = distill_tool_result(original)
    assert clone["ok"] is False and clone["error"] == "timeout"


def test_distill_tool_result_noops_on_non_string_or_missing_field():
    assert distill_tool_result({"tool": "x", "ok": True}) == {"tool": "x", "ok": True}
    assert distill_tool_result({"output": 123}) == {"output": 123}
    assert distill_tool_result("not a dict") == "not a dict"


# --------------------------------------------------------------------------- #
# Profile selection
# --------------------------------------------------------------------------- #

def test_profile_for_sniffs_content_over_tool_name():
    assert profile_for("browser_open", NOISY_JSON).name == "json_response"  # JSON content wins
    assert profile_for("unknown_tool", NOISY_HTML).name == "browser_html"
    assert profile_for("ci_build_logs", "plain line\nanother").name == "log_dump"
    assert profile_for(None, "nothing special").name == "generic"


def test_all_committed_profiles_load():
    assert {"generic", "browser_html", "log_dump", "tool_trace", "json_response"} <= set(PROFILES)
