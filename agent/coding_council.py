# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Data-driven coding council routing for Sophia."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agent.config import ROOT

COUNCIL_PATH = ROOT / "data" / "coding_council_figures.json"


def load_coding_council(path: Path = COUNCIL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _term_matches(text: str, term: str) -> bool:
    normalized = text.lower()
    needle = term.lower()
    if needle.strip() != needle:
        return needle in f" {normalized} "
    if re.search(r"[^a-zA-Z0-9_+#.-]", needle):
        return needle in normalized
    return bool(re.search(rf"(?<![a-zA-Z0-9_]){re.escape(needle)}(?![a-zA-Z0-9_])", normalized))


def _score_seat(text: str, seat: dict[str, Any]) -> int:
    return sum(1 for term in seat.get("triggerTerms", []) if _term_matches(text, str(term)))


def _select_ranked(group: dict[str, Any], text: str, *, limit: int) -> list[dict[str, Any]]:
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for key, seat in group.items():
        if not isinstance(seat, dict):
            continue
        score = _score_seat(text, seat)
        if score:
            ranked.append((score, key, seat))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [seat for _, _, seat in ranked[:limit]]


def route_coding_council(
    prompt: str,
    materials: list[str] | None = None,
    *,
    max_language_seats: int = 3,
    max_role_seats: int = 5,
    max_platform_seats: int = 2,
    max_specialist_seats: int = 5,
    max_improvement_seats: int = 3,
) -> dict[str, Any]:
    """Select council seats relevant to a coding task."""
    data = load_coding_council()
    text = f"{prompt}\n" + "\n".join(materials or [])
    language_seats = _select_ranked(data.get("languageElders", {}), text, limit=max_language_seats)
    role_seats = _select_ranked(data.get("expertRoles", {}), text, limit=max_role_seats)
    platform_seats = _select_ranked(data.get("platformExperts", {}), text, limit=max_platform_seats)
    specialist_seats = _select_ranked(
        data.get("engineeringSpecialists", {}),
        text,
        limit=max_specialist_seats,
    )
    improvement_seats = _select_ranked(
        data.get("improvementWriters", {}),
        text,
        limit=max_improvement_seats,
    )

    role_ids = {seat.get("seatId") for seat in role_seats}
    for seat_id in data.get("workflow", {}).get("defaultSeats", []):
        if seat_id in role_ids:
            continue
        seat = next(
            (
                candidate
                for candidate in data.get("expertRoles", {}).values()
                if isinstance(candidate, dict) and candidate.get("seatId") == seat_id
            ),
            None,
        )
        if seat:
            role_seats.append(seat)
            role_ids.add(seat_id)
        if len(role_seats) >= max_role_seats:
            break

    specialist_ids = {seat.get("seatId") for seat in specialist_seats}
    for seat_id in data.get("workflow", {}).get("defaultSpecialistSeats", []):
        if seat_id in specialist_ids:
            continue
        seat = next(
            (
                candidate
                for candidate in data.get("engineeringSpecialists", {}).values()
                if isinstance(candidate, dict) and candidate.get("seatId") == seat_id
            ),
            None,
        )
        if seat:
            specialist_seats.append(seat)
            specialist_ids.add(seat_id)
        if len(specialist_seats) >= max_specialist_seats:
            break

    if not language_seats:
        language_seats = [data.get("workflow", {}).get("fallbackLanguageSeat", {})]

    return {
        "languageSeats": language_seats,
        "roleSeats": role_seats,
        "platformSeats": platform_seats,
        "specialistSeats": specialist_seats,
        "improvementSeats": improvement_seats,
        "decisionContract": data.get("workflow", {}).get("decisionContract", []),
    }


def format_coding_council(route: dict[str, Any]) -> str:
    """Format selected seats as compact prompt context."""
    lines = [
        "## Coding Council",
        "Use source-inspired engineering seats, not impersonation. Each seat contributes constraints only.",
    ]
    for label, key in (
        ("Language seats", "languageSeats"),
        ("Role seats", "roleSeats"),
        ("Platform seats", "platformSeats"),
        ("Engineering specialist seats", "specialistSeats"),
        ("Self-improvement and writing-method seats", "improvementSeats"),
    ):
        seats = [seat for seat in route.get(key, []) if isinstance(seat, dict) and seat]
        if not seats:
            continue
        lines.append(f"\n### {label}")
        for seat in seats:
            emphasis = "; ".join(str(item) for item in seat.get("decisionEmphasis", [])[:4])
            boundary = seat.get("speakerBoundary")
            frame = seat.get("representativeFrame") or seat.get("sourceFrame")
            details = []
            if frame:
                details.append(str(frame))
            if boundary:
                details.append(str(boundary))
            if emphasis:
                details.append(f"Emphasis: {emphasis}.")
            detail_text = " ".join(details)
            lines.append(f"- {seat.get('displayName', seat.get('seatId', 'Council seat'))}: {detail_text}")
    if route.get("decisionContract"):
        lines.append("\n### Council decision contract")
        for item in route["decisionContract"]:
            lines.append(f"- {item}")
    return "\n".join(lines)
