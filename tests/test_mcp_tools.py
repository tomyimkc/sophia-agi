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


def main() -> int:
    test_validate_ok()
    test_get_attribution_ddj()
    test_benchmark_list_philosophy()
    test_gate_rejects_bad_ddj()
    test_list_disputes()
    test_export_corpus()
    test_web_evidence_search_offline()
    test_rubric_review_flags_missing()
    print("test_mcp_tools: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
