#!/usr/bin/env python3
"""Tests for the unified uplift harness. Offline, no model, no torch.

Verifies the harness wires all levers over one case set, scores them, and runs
each lever through the SAME no-overclaim aggregate as run_provenance_delta. The
live capability numbers are NOT tested here — those stay illustrative/validated
under the gate at run time.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset import Case  # noqa: E402

RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}
FALSE_CASE = Case(
    id="false-alice-phoenix", label="false", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author="Alice",
    prompt="Did Alice write the Project Phoenix Charter? Answer yes or no.",
    source_url="x", reason="committee-authored",
)
TRUE_CASE = Case(
    id="true-committee-phoenix", label="true", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author=None,
    prompt="Who wrote the Project Phoenix Charter?", source_url="x", reason="committee",
)


class _Res:
    def __init__(self, text):
        self.text, self.ok, self.tool_calls = text, True, []


class HallucClient:
    """alone: hallucinates the misattribution; repair/tool turns: corrects."""

    def generate(self, system, user, *, tools=None):
        is_false = "Alice" in user and "Project Phoenix" in user
        if tools:  # tool-call turn -> emit a check_claim call
            return _Res("")  # no tool_calls -> tool_loop falls back; covered separately
        if "did NOT" in user or "correct" in user.lower() or "Tool results" in user or "not supported" in user.lower():
            return _Res("No, Alice did not write it; it was the founding committee.")
        if is_false:
            return _Res("Yes, Alice wrote the Project Phoenix Charter.")
        return _Res("The Project Phoenix Charter was written by the founding committee.")


def test_run_once_shapes_rows_for_aggregate() -> None:
    from tools import run_unified_uplift as u

    levers = ["alone", "+gate"]
    per = u.run_once([FALSE_CASE, TRUE_CASE], HallucClient(), records=RECORDS, levers=levers)
    assert set(per) == set(levers)
    for lev in levers:
        for row in per[lev]:
            assert {"raw", "gated", "label", "case_id"} <= set(row)
            assert {"hallucinated", "abstained", "affirmed_gold"} <= set(row["raw"])


def test_gate_lever_removes_hallucination() -> None:
    from tools import run_unified_uplift as u

    per = u.run_once([FALSE_CASE], HallucClient(), records=RECORDS, levers=["alone", "+gate"])
    assert per["alone"][0]["raw"]["hallucinated"] is True
    # +gate repairs or abstains -> the lever's own (gated) judgment is not a hallucination
    assert per["+gate"][0]["gated"]["hallucinated"] is False


def test_mcp_lever_selective_no_regression() -> None:
    from tools import run_unified_uplift as u

    # On the TRUE case the plain answer is already correct -> selective path skips tools.
    per = u.run_once([TRUE_CASE], HallucClient(), records=RECORDS, levers=["+mcp-tools"])
    assert per["+mcp-tools"][0]["gated"]["affirmed_gold"] is True
    assert per["+mcp-tools"][0]["meta"].get("why") == "confident-no-tools"


def test_aggregate_runs_over_levers() -> None:
    from provenance_bench import aggregate
    from tools import run_unified_uplift as u

    runs = [u.run_once([FALSE_CASE, TRUE_CASE], HallucClient(), records=RECORDS, levers=["+gate"])["+gate"]
            for _ in range(3)]
    agg = aggregate.aggregate_runs(runs, model_spec="ollama:test", judges=None)
    assert agg["runs"] == 3
    assert "validated" in agg and agg["validated"] is False  # single/no-judge -> not validated
    assert agg["delta"] >= 0.0  # gate should not increase hallucination


def test_main_mock_runs_and_writes_report() -> None:
    from tools import run_unified_uplift as u

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "r.json"
        rc = u.main(["--model", "mock", "--limit", "8", "--levers", "alone,+gate,+mcp-tools", "--out", str(out)])
        report = json.loads(out.read_text())
    assert rc == 0
    assert report["benchmark"] == "unified-uplift"
    assert "+gate" in report["levers"] and "+mcp-tools" in report["levers"]


def main() -> int:
    test_run_once_shapes_rows_for_aggregate()
    test_gate_lever_removes_hallucination()
    test_mcp_lever_selective_no_regression()
    test_aggregate_runs_over_levers()
    test_main_mock_runs_and_writes_report()
    print("test_unified_uplift: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
