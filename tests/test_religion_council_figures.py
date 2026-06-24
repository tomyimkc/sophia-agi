#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke tests for religion figure council data and prompt wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prompts import SHARED_RULES  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_figure_seats_exist() -> None:
    data = load_json(ROOT / "data" / "religion_council_figures.json")
    assert "jesus_of_nazareth" in data
    assert "gautama_buddha" in data
    assert data["jesus_of_nazareth"]["seatId"] == "jesus_gospel_witness"
    assert data["gautama_buddha"]["seatId"] == "buddhist_dharma_witness"


def test_no_impersonation_boundary() -> None:
    data = load_json(ROOT / "data" / "religion_council_figures.json")
    for record in data.values():
        boundary = record["speakerBoundary"].lower()
        assert "do not impersonate" in boundary
        assert "source" in boundary or "tradition" in boundary


def test_prompt_mentions_religion_figure_council() -> None:
    lowered = SHARED_RULES.lower()
    assert "religion figure source council" in lowered
    assert "do not speak in first person" in lowered
    assert "jesus tradition witness" in lowered
    assert "buddhist dharma witness" in lowered


def main() -> int:
    test_figure_seats_exist()
    test_no_impersonation_boundary()
    test_prompt_mentions_religion_figure_council()
    print("test_religion_council_figures: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
