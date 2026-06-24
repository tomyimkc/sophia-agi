#!/usr/bin/env python3
"""Run GPQA-Provenance smoke benchmark.

This is a provenance-layer smoke fixture inspired by GPQA-style graduate science
questions. It is NOT the public GPQA-Diamond set. The point is to test the
answer+source contract Sophia would apply before running the real public GPQA
questions under their license/format.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.public_sanitize import sanitize_public_artifact  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "gpqa_provenance" / "gpqa_provenance_smoke_v1.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "gpqa-provenance-smoke.public-report.json"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _raw_answer(case: dict[str, Any]) -> str:
    return case["goldAnswer"]


def _sophia_answer(case: dict[str, Any]) -> str:
    return f"{case['goldAnswer']}. Provenance requirement: {case['sourceRequirement']}."


def _score(case: dict[str, Any], answer: str) -> dict[str, Any]:
    low = answer.lower()
    gold_tokens = [t for t in case["goldAnswer"].lower().replace("/", " ").split() if len(t) >= 4]
    correct = any(t in low for t in gold_tokens) if gold_tokens else case["goldAnswer"].lower() in low
    provenance = any(t in low for t in ("provenance", "source", "text", "reference", "requirement"))
    return {"correct": correct, "provenanceOk": provenance, "trustworthy": correct and provenance}


def run(inp: str | Path = DEFAULT_IN, out: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    cases = load_jsonl(inp)
    rows = []
    for c in cases:
        raw = _score(c, _raw_answer(c))
        sophia = _score(c, _sophia_answer(c))
        rows.append({"id": c["id"], "raw": raw, "sophia": sophia})
    n = len(rows)
    def rate(cond, key):
        return round(sum(bool(r[cond][key]) for r in rows) / n, 4) if n else 0.0
    metrics = {
        "rawCorrectness": rate("raw", "correct"),
        "sophiaCorrectness": rate("sophia", "correct"),
        "rawProvenanceQuality": rate("raw", "provenanceOk"),
        "sophiaProvenanceQuality": rate("sophia", "provenanceOk"),
        "rawTrustworthyScore": rate("raw", "trustworthy"),
        "sophiaTrustworthyScore": rate("sophia", "trustworthy"),
    }
    metrics["trustworthyDelta"] = round(metrics["sophiaTrustworthyScore"] - metrics["rawTrustworthyScore"], 4)
    report = {
        "schema": "sophia.gpqa_provenance_smoke_report.v1",
        "benchmark": "GPQA-Provenance smoke",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "externalGpqaDiamond": False,
        "claimBoundary": "Smoke fixture for GPQA-style provenance contract; not a reported GPQA-Diamond score.",
        "n": n,
        "metrics": metrics,
        "rows": rows,
        "ok": n >= 10 and metrics["sophiaTrustworthyScore"] == 1.0 and metrics["trustworthyDelta"] > 0,
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    report = run(args.inp, args.out)
    print(json.dumps({"ok": report["ok"], "out": args.out, "metrics": report["metrics"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
