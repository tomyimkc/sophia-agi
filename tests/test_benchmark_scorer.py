#!/usr/bin/env python3
"""Regression tests for benchmark heuristic scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case  # noqa: E402

RUNS = ROOT / "benchmark" / "model_runs"
TRADITIONS = ROOT / "data" / "traditions.json"


def _case(domain: str, case_id: str) -> dict:
    bench = load_json(DOMAIN_BENCH[domain])
    for case in bench["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def test_sophia_v1_philosophy_deny_heuristics() -> None:
    traditions = load_json(TRADITIONS)
    payload = json.loads((RUNS / "local-sophia-v1-philosophy.json").read_text(encoding="utf-8"))
    responses = payload["responses"]
    for case_id in ("trap_confucius_ddj", "symposium_dialogue_not_autograph"):
        case = _case("philosophy", case_id)
        ok, reasons = score_case(case, responses[case_id], traditions)
        assert ok, f"{case_id} should pass after deny heuristic fix: {reasons}"


def test_sophia_v1_stockholm_still_needs_pop_myth() -> None:
    traditions = load_json(TRADITIONS)
    payload = json.loads((RUNS / "local-sophia-v1-psychology.json").read_text(encoding="utf-8"))
    case = _case("psychology", "stockholm_every_kidnapping")
    ok, reasons = score_case(case, payload["responses"]["stockholm_every_kidnapping"], traditions)
    assert not ok
    assert any("pop_myth" in r for r in reasons)


def main() -> int:
    test_sophia_v1_philosophy_deny_heuristics()
    test_sophia_v1_stockholm_still_needs_pop_myth()
    print("test_benchmark_scorer: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())