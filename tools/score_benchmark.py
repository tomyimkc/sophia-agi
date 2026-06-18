#!/usr/bin/env python3
"""Score model responses against Sophia AGI per-domain benchmarks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case  # noqa: E402


def load_responses(payload: dict) -> dict[str, str]:
    if "responses" in payload and isinstance(payload["responses"], dict):
        return {k: str(v) for k, v in payload["responses"].items()}
    return {k: str(v) for k, v in payload.items() if k not in {"model", "date", "domain"}}


def score_all(responses: dict, bench: dict, traditions: dict) -> dict:
    cases = bench.get("cases", [])
    results = []
    passed = 0
    for case in cases:
        case_id = case["id"]
        response = responses.get(case_id, "")
        ok, reasons = score_case(case, response, traditions)
        if ok:
            passed += 1
        results.append({"id": case_id, "passed": ok, "reasons": reasons})
    total = len(cases)
    return {
        "domain": bench.get("domain", "unknown"),
        "version": bench.get("version", 1),
        "passed": passed,
        "total": total,
        "score_pct": round(100.0 * passed / total, 1) if total else 0.0,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Sophia AGI benchmark responses")
    parser.add_argument("responses", type=Path, help="Responses JSON file")
    parser.add_argument("--domain", choices=list(DOMAIN_BENCH.keys()), default="philosophy")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payload = load_json(args.responses)
    domain = args.domain or payload.get("domain", "philosophy")
    bench = load_json(DOMAIN_BENCH[domain])
    traditions = load_json(ROOT / "data" / "traditions.json")
    report = score_all(load_responses(payload), bench, traditions)
    if payload.get("model"):
        report["model"] = payload["model"]

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())