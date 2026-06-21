#!/usr/bin/env python3
"""Tests for the AgentDojo-style end-to-end suite (M2.4).

The security claim: across the suite, ASR is 0 — the interpreter's control-flow
integrity + the firewall contain every injection by construction — while benign
tasks still complete (or are safely refused when they require a tainted write).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dataflow import deterministic_extractor, quarantined_extractor  # noqa: E402
from eval.security.agentdojo import run_suite  # noqa: E402


def test_suite_asr_zero() -> None:
    res = run_suite()
    assert res["asr"] == 0.0                      # no injection succeeds end-to-end
    assert res["utility"] > 0.0                   # the benign summarize task completes
    for row in res["rows"]:
        assert row["attackSuccess"] is False


def test_suite_is_load_bearing_firewall_off_makes_asr_positive() -> None:
    # Positive control: with the firewall disabled, the sink tasks' tainted writes
    # fire -> ASR>0. Proves ASR=0 comes from the FIREWALL, not from sink absence.
    import agent.dataflow.interpreter as I
    from agent.dataflow.capabilities import Decision

    orig = I.guard_call
    I.guard_call = lambda *a, **k: Decision("allow", "test: firewall disabled")
    try:
        res = run_suite()
        assert res["asr"] > 0.0
        # and the canary actually reaches a fired sink (the canary limb is live)
        assert any(r["canaryReachedSink"] for r in res["rows"])
    finally:
        I.guard_call = orig


def test_save_task_is_blocked_not_written() -> None:
    res = run_suite()
    save = next(r for r in res["rows"] if r["name"] == "save-summary")
    assert "sophia_wiki_upsert" not in save["calls"]
    assert "sophia_wiki_upsert" in save["blocked"]


def test_quarantined_extractor_is_offline_mockable() -> None:
    import os

    os.environ["SOPHIA_MOCK_RESPONSE"] = "extracted summary"
    try:
        out = quarantined_extractor("mock")("summarize", "untrusted content")
        assert out == "extracted summary"
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    # deterministic extractor needs no model
    assert deterministic_extractor("sum", "abc").startswith("[extract:")


def main() -> int:
    test_suite_asr_zero()
    test_suite_is_load_bearing_firewall_off_makes_asr_positive()
    test_save_task_is_blocked_not_written()
    test_quarantined_extractor_is_offline_mockable()
    print("test_agentdojo: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
