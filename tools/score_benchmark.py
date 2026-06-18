#!/usr/bin/env python3
"""Score model responses against the Sophia Attribution Benchmark."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "tests" / "attribution_bench.json"
ATTRIBUTIONS_PATH = ROOT / "data" / "attributions.json"

DENY_PATTERNS = [
    r"\bdid not\b",
    r"\bdidn't\b",
    r"\bnot write\b",
    r"\bfalse\b",
    r"\bincorrect\b",
    r"\bno[,.]",
    r"並未",
    r"沒有",
    r"不是",
]
UNCERTAIN_PATTERNS = {
    "compiled": [r"compiled", r"assembled", r"students", r"editors", r"彙編", r"編纂", r"弟子"],
    "legendary": [
        r"uncertain",
        r"disputed",
        r"legendary",
        r"debated",
        r"存疑",
        r"不確定",
        r"疑",
    ],
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return text.lower().strip()


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def load_responses(payload: dict) -> dict[str, str]:
    if "responses" in payload and isinstance(payload["responses"], dict):
        return {k: str(v) for k, v in payload["responses"].items()}
    return {k: str(v) for k, v in payload.items() if k not in {"model", "date"}}


def tradition_markers(tradition_id: str, traditions: dict) -> list[str]:
    record = traditions.get(tradition_id, {})
    markers = [tradition_id, tradition_id.replace("_", " ")]
    for key in ("labelEn", "labelZh"):
        value = record.get(key)
        if value:
            markers.append(str(value).lower())
    return markers


def score_case(
    case: dict,
    response: str,
    attributions: dict,
    traditions: dict,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    text = normalize(response)
    ok = True

    deny = case.get("mustDenyAttribution")
    if deny:
        author = deny["author"]
        if not matches_any(text, DENY_PATTERNS):
            ok = False
            reasons.append(f"expected denial of {author} attribution")

    affirm = case.get("mustAffirmAuthor")
    if affirm:
        author = affirm["author"]
        if author not in text:
            ok = False
            reasons.append(f"expected mention of correct author '{author}'")

    signal = case.get("mustSignalConfidence")
    if signal:
        confidence = signal["confidence"]
        patterns = UNCERTAIN_PATTERNS.get(confidence, [confidence])
        if not matches_any(text, patterns):
            ok = False
            reasons.append(f"expected signal for confidence '{confidence}'")

    for tradition in case.get("mustMentionTraditions", []):
        markers = tradition_markers(tradition, traditions)
        if not any(marker.lower() in text for marker in markers):
            ok = False
            reasons.append(f"expected tradition context '{tradition}'")

    return ok, reasons


def score_all(responses: dict, bench: dict, attributions: dict, traditions: dict) -> dict:
    cases = bench.get("cases", [])
    results = []
    passed = 0
    for case in cases:
        case_id = case["id"]
        response = responses.get(case_id, "")
        ok, reasons = score_case(case, response, attributions, traditions)
        if ok:
            passed += 1
        results.append({
            "id": case_id,
            "passed": ok,
            "reasons": reasons,
        })
    total = len(cases)
    return {
        "version": bench.get("version", 1),
        "passed": passed,
        "total": total,
        "score_pct": round(100.0 * passed / total, 1) if total else 0.0,
        "results": results,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tools/score_benchmark.py <responses.json> [--out results.json]")
        return 1

    responses_path = Path(sys.argv[1])
    out_path = None
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])

    payload = load_json(responses_path)
    responses = load_responses(payload)
    bench = load_json(BENCH_PATH)
    attributions = load_json(ATTRIBUTIONS_PATH)
    traditions = load_json(ROOT / "data" / "traditions.json")
    report = score_all(responses, bench, attributions, traditions)
    if payload.get("model"):
        report["model"] = payload["model"]

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())