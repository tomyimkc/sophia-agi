#!/usr/bin/env python3
"""Smoke tests for sophia_mcp tool implementations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp.tools_impl import (  # noqa: E402
    benchmark_list,
    export_corpus,
    gate_check,
    get_attribution,
    list_disputes,
    rubric_review,
    sector_council,
    validate_corpus,
    web_evidence_search,
)


def test_validate_ok() -> None:
    result = validate_corpus()
    assert result["ok"] is True
    assert result["trainingExamples"] >= 500


def test_get_attribution_ddj() -> None:
    record = get_attribution("dao_de_jing")
    assert record["textId"] == "dao_de_jing"
    assert "confucius" in record["doNotAttributeTo"]


def test_benchmark_list_philosophy() -> None:
    data = benchmark_list("philosophy")
    assert len(data["cases"]) == 9


def test_gate_rejects_bad_ddj() -> None:
    bad = "Yes, Confucius wrote the Dao De Jing."
    result = gate_check(bad, "Did Confucius write the Dao De Jing?", mode="advisor")
    assert result["passed"] is False


def test_list_disputes() -> None:
    data = list_disputes()
    assert data["count"] >= 10


def test_export_corpus() -> None:
    result = export_corpus()
    assert result["ok"] is True
    assert result["lines"] >= 500


def test_web_evidence_search_offline() -> None:
    result = web_evidence_search("Dao De Jing attribution", online=False, local_top_k=1)
    assert result["web"]["online"] is False
    assert result["localSources"]


def test_rubric_review_flags_missing() -> None:
    result = rubric_review(
        "Should the answer include sources?",
        "Decision: yes.\n中文摘要: 是。",
        must_include=["source path"],
    )
    assert result["strictPassReady"] is False
    assert result["missing"]


def test_sector_council_law_routes_specialist() -> None:
    result = sector_council("law", "gacha odds disclosure and refund policy for a Hong Kong game launch")
    assert result["councilId"] == "law"
    assert "gaming_monetization_lawyer_seat" in result["seatedSeatIds"]
    assert "human_review_gatekeeper_seat" in result["seatedSeatIds"]  # guardian
    assert result["humanBoundary"]
    assert "not" in result["notAdvice"].lower()


def test_sector_council_auto_detects() -> None:
    result = sector_council("auto", "set up KYC and AML for our Stripe payment processor")
    assert result["councilId"] == "financial"
    assert "aml_kyc_seat" in result["seatedSeatIds"]


def test_sector_council_rejects_unknown() -> None:
    assert "error" in sector_council("medical", "diagnose this")


def main() -> int:
    test_validate_ok()
    test_get_attribution_ddj()
    test_benchmark_list_philosophy()
    test_gate_rejects_bad_ddj()
    test_list_disputes()
    test_export_corpus()
    test_web_evidence_search_offline()
    test_rubric_review_flags_missing()
    test_sector_council_law_routes_specialist()
    test_sector_council_auto_detects()
    test_sector_council_rejects_unknown()
    print("test_mcp_tools: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
