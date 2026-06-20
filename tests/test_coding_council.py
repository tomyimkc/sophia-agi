#!/usr/bin/env python3
"""Smoke tests for Sophia's coding council router."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.coding_council import format_coding_council, load_coding_council, route_coding_council  # noqa: E402
from agent.prompts import SHARED_RULES  # noqa: E402


def test_coding_council_data_loads() -> None:
    data = load_coding_council()
    assert "languageElders" in data
    assert "expertRoles" in data
    assert "platformExperts" in data
    assert data["languageElders"]["python"]["speakerBoundary"]


def test_router_selects_language_role_and_platform() -> None:
    route = route_coding_council(
        "Fix a Python FastAPI bug on Linux and add pytest regression tests.",
        ["api/server.py", "tests/test_api.py"],
    )
    language_names = {seat["displayName"] for seat in route["languageSeats"]}
    role_names = {seat["displayName"] for seat in route["roleSeats"]}
    platform_names = {seat["displayName"] for seat in route["platformSeats"]}
    specialist_names = {seat["displayName"] for seat in route["specialistSeats"]}
    assert "Python readability seat" in language_names
    assert "QA and test reviewer" in role_names
    assert "Linux platform reviewer" in platform_names
    assert "Tool-calling reliability engineer" in specialist_names


def test_formatted_council_mentions_no_impersonation() -> None:
    route = route_coding_council("Implement a TypeScript React UI fix.", [])
    formatted = format_coding_council(route)
    assert "not impersonation" in formatted
    assert "TypeScript type-safety seat" in formatted
    assert "Frontend engineering reviewer" in formatted


def test_router_selects_rag_eval_and_improvement_writers() -> None:
    route = route_coding_council(
        "Build a RAG writing workflow to improve strict pass hidden test scores with small habits.",
        ["retrieval.py", "hidden_eval_protocol.py"],
    )
    specialist_names = {seat["displayName"] for seat in route["specialistSeats"]}
    improvement_names = {seat["displayName"] for seat in route["improvementSeats"]}
    assert "RAG/context engineering reviewer" in specialist_names
    assert "Evaluation and benchmark engineer" in specialist_names
    assert "Atomic Habits method seat" in improvement_names


def test_shared_prompt_mentions_coding_council() -> None:
    lowered = SHARED_RULES.lower()
    assert "coding council" in lowered
    assert "not impersonation" in lowered


def main() -> int:
    test_coding_council_data_loads()
    test_router_selects_language_role_and_platform()
    test_formatted_council_mentions_no_impersonation()
    test_router_selects_rag_eval_and_improvement_writers()
    test_shared_prompt_mentions_coding_council()
    print("test_coding_council: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
