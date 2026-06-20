#!/usr/bin/env python3
"""Tests for the law / financial / economy sector councils."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import sector_council as sc  # noqa: E402

COUNCILS = ["law", "financial", "economy"]


def _all_seats(council: dict) -> list[dict]:
    return [seat for group in council["seatGroups"].values() for seat in group["seats"].values()]


def test_three_councils_load() -> None:
    assert set(sc.available_councils()) == set(COUNCILS)
    for cid in COUNCILS:
        council = sc.load_council(cid)
        assert council["councilId"] == cid
        assert council.get("displayName")
        assert council.get("humanBoundary")
        assert council["workflow"]["decisionContract"]


def test_seat_schema_and_unique_ids() -> None:
    for cid in COUNCILS:
        council = sc.load_council(cid)
        seats = _all_seats(council)
        seat_ids = [s["seatId"] for s in seats]
        # within a council, non-shared seatIds should be unique (guardians may repeat
        # the same id across councils, but not within one council)
        assert len(seat_ids) == len(set(seat_ids)), f"duplicate seatId within {cid}"
        for seat in seats:
            assert seat.get("seatId") and seat.get("displayName")
            assert "decisionEmphasis" in seat


def test_figure_seats_have_speaker_boundary() -> None:
    # any seat that names a source lineage must forbid impersonation
    for cid in COUNCILS:
        for seat in _all_seats(sc.load_council(cid)):
            frame = (seat.get("sourceFrame") or "")
            if "lineage" in frame.lower() or "witness" in frame.lower():
                assert seat.get("speakerBoundary"), f"{cid}:{seat['seatId']} names a lineage but has no speakerBoundary"


def test_guardians_always_seated() -> None:
    # even an empty / irrelevant query must seat every core guardian + defaults
    for cid in COUNCILS:
        council = sc.load_council(cid)
        route = sc.route_council(council, "completely unrelated gibberish zzz")
        present = {s["seatId"] for g in route["selected"].values() for s in g["seats"]}
        for default_id in council["workflow"]["defaultSeats"]:
            assert default_id in present, f"{cid}: default seat {default_id} not seated"


def test_law_routes_gaming_and_jurisdiction() -> None:
    council = sc.load_council("law")
    route = sc.route_council(council, "Review our gacha odds disclosure and virtual currency rules for Hong Kong")
    present = {s["seatId"] for g in route["selected"].values() for s in g["seats"]}
    assert "gaming_monetization_lawyer_seat" in present
    assert "hong_kong_jurisdiction_seat" in present
    assert "human_review_gatekeeper_seat" in present  # guardian


def test_financial_routes_aml_and_payments() -> None:
    council = sc.load_council("financial")
    route = sc.route_council(council, "Set up KYC and AML for our Stripe payment processor onboarding")
    present = {s["seatId"] for g in route["selected"].values() for s in g["seats"]}
    assert "aml_kyc_seat" in present
    assert "payments_processor_seat" in present
    assert "numbers_auditor_seat" in present  # guardian


def test_economy_routes_labor_and_stakeholders() -> None:
    council = sc.load_council("economy")
    route = sc.route_council(council, "Simulate a minimum wage increase and its effect on workers and small business")
    present = {s["seatId"] for g in route["selected"].values() for s in g["seats"]}
    assert "labor_economist_seat" in present
    assert ("worker_impact_seat" in present) or ("small_business_impact_seat" in present)
    assert "value_judgment_flagger_seat" in present  # guardian


def test_fallback_when_no_specialist_matches() -> None:
    council = sc.load_council("law")
    route = sc.route_council(council, "zzz nothing matches here zzz")
    present = {s["seatId"] for g in route["selected"].values() for s in g["seats"]}
    assert "general_legal_reviewer_seat" in present


def test_format_council_renders_boundary_and_contract() -> None:
    council = sc.load_council("financial")
    text = sc.format_council(sc.route_council(council, "valuation of a startup with a DCF"))
    assert "Human authority boundary" in text
    assert "Council decision contract" in text
    assert "not financial" in text.lower() or "not advice" in text.lower() or "中文摘要" in text


def test_detect_council_picks_right_sector() -> None:
    assert sc.detect_council("review our contract and tax compliance for the EU") == "law"
    assert sc.detect_council("model runway, WACC, and AML for our payment processor") == "financial"
    assert sc.detect_council("simulate the inflation and fiscal-deficit tradeoff") == "economy"
    # a non-sector question should not convene any council
    assert sc.detect_council("did Confucius write the Dao De Jing?") is None


def test_agent_prompt_injects_council_block() -> None:
    # the agent build_user_prompt should inject the sector-council block for a
    # legal question; stub the heavy IO so the test is fast and hermetic
    from tools import sophia_agent as agent

    class FakeChunk:
        path = "data/x.json"
        title = "stub"
        excerpt = "stub excerpt"
        score = 0.5

    saved = (agent.retrieve, agent.gather_evidence, agent.recent_decisions)
    agent.retrieve = lambda q, *, top_k=8: [FakeChunk()]
    agent.gather_evidence = lambda q, **k: {"web": {"online": False, "sources": []}}
    agent.recent_decisions = lambda limit=3: []
    try:
        prompt = agent.build_user_prompt("advisor", "Review our gacha odds disclosure and refund policy for Hong Kong")
    finally:
        agent.retrieve, agent.gather_evidence, agent.recent_decisions = saved
    assert "Law & Governance Council" in prompt
    assert "Human authority boundary" in prompt


def main() -> int:
    test_three_councils_load()
    test_seat_schema_and_unique_ids()
    test_figure_seats_have_speaker_boundary()
    test_guardians_always_seated()
    test_law_routes_gaming_and_jurisdiction()
    test_financial_routes_aml_and_payments()
    test_economy_routes_labor_and_stakeholders()
    test_fallback_when_no_specialist_matches()
    test_format_council_renders_boundary_and_contract()
    test_detect_council_picks_right_sector()
    test_agent_prompt_injects_council_block()
    print("test_sector_councils: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
