#!/usr/bin/env python3
"""Score curated RAG + LLM path on all 23 benchmark cases.

Usage:
  python tools/eval_rag_benchmark.py --dry-run
  python tools/eval_rag_benchmark.py
  python tools/eval_rag_benchmark.py --cases stockholm_every_kidnapping ancestor_veneration_split buddha_nirvana_pop
  python tools/eval_rag_benchmark.py --backend keyword
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case  # noqa: E402
from agent.config import load_dotenv  # noqa: E402
from agent.rag_pipeline import answer_question  # noqa: E402

OUT_DIR = ROOT / "benchmark" / "model_runs"


def all_cases() -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    for domain, path in DOMAIN_BENCH.items():
        bench = load_json(path)
        for case in bench.get("cases", []):
            items.append((domain, case))
    return items


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate Sophia online RAG on benchmark cases")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", default="", help="Set SOPHIA_RAG_BACKEND for this run")
    parser.add_argument("--cases", nargs="*", help="Subset of case ids")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.backend:
        import os

        os.environ["SOPHIA_RAG_BACKEND"] = args.backend

    cases = all_cases()
    if args.cases:
        wanted = set(args.cases)
        cases = [(d, c) for d, c in cases if c["id"] in wanted]

    print(f"Cases: {len(cases)}")
    if args.dry_run:
        for domain, case in cases:
            print(f"  {domain}/{case['id']}")
        return 0

    traditions = load_json(ROOT / "data" / "traditions.json")
    by_domain: dict[str, dict] = {}
    results_all: list[dict] = []

    for domain, case in cases:
        print(f"RAG / {domain} / {case['id']}...")
        result = answer_question(case["question"], mode="advisor", top_k=8)
        answer = result["answer"]
        ok, reasons = score_case(case, answer, traditions)
        gate = result.get("gate", {})
        row = {
            "id": case["id"],
            "domain": domain,
            "passed": ok,
            "reasons": reasons,
            "gatePassed": gate.get("passed"),
            "sources": [s["path"] for s in result.get("sources", [])[:3]],
        }
        results_all.append(row)
        bucket = by_domain.setdefault(domain, {"responses": {}, "results": []})
        bucket["responses"][case["id"]] = answer
        bucket["results"].append(row)

    passed = sum(1 for r in results_all if r["passed"])
    total = len(results_all)
    print(f"\nRAG benchmark: {passed}/{total} ({100 * passed / total:.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    backend = args.backend or __import__("os").environ.get("SOPHIA_RAG_BACKEND", "auto")

    for domain, payload in by_domain.items():
        run = {
            "domain": domain,
            "model": f"rag-{backend}",
            "date": stamp,
            "responses": payload["responses"],
        }
        out = args.out or (OUT_DIR / f"rag-{backend}-{domain}.json")
        if len(by_domain) > 1:
            out = OUT_DIR / f"rag-{backend}-{domain}.json"
        out.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        domain_passed = sum(1 for r in payload["results"] if r["passed"])
        domain_total = len(payload["results"])
        report = {
            "domain": domain,
            "model": f"rag-{backend}",
            "passed": domain_passed,
            "total": domain_total,
            "score_pct": round(100.0 * domain_passed / domain_total, 1) if domain_total else 0.0,
            "results": payload["results"],
        }
        report_path = out.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out}")

    summary_path = OUT_DIR / f"rag-{backend}-summary.json"
    summary_path.write_text(
        json.dumps({"passed": passed, "total": total, "results": results_all}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())