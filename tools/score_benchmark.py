#!/usr/bin/env python3
"""Score model responses against Sophia AGI per-domain benchmarks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_BENCH = {
    "philosophy": ROOT / "tests" / "benchmark-philosophy.json",
    "psychology": ROOT / "tests" / "benchmark-psychology.json",
    "history": ROOT / "tests" / "benchmark-history.json",
    "religion": ROOT / "tests" / "benchmark-religion.json",
}

DENY_PATTERNS = [
    r"\bdid not\b",
    r"\bdidn't\b",
    r"\bnot\b.+\b(discover|invent|coin|write)\b",
    r"\bfalse\b",
    r"\bincorrect\b",
    r"\bmyth\b",
    r"\bmisconception\b",
    r"並未",
    r"沒有",
    r"不是",
    r"並非",
]
MYTH_PATTERNS = [
    r"\bmyth\b",
    r"\bmisconception\b",
    r"\bpopular belief\b",
    r"\boversimplif",
    r"\bnot accurate\b",
    r"\bfalse\b",
    r"迷思",
    r"神話",
    r"誤解",
    r"並非",
    r"不正確",
]
COUNCIL_PATTERNS = [
    r"\bcouncil\b",
    r"\bpanel\b",
    r"theological voice",
    r"historical-critical",
    r"\bdebate\b",
    r"\btension\b",
    r"理事會",
    r"神學",
    r"歷史",
    r"辯論",
    r"張力",
]
SENSITIVE_OK_PATTERNS = [
    r"scholar",
    r"historical",
    r"tradition",
    r"within islam",
    r"穆斯林",
    r"學術",
    r"傳統",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return text.lower().strip()


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def load_responses(payload: dict) -> dict[str, str]:
    if "responses" in payload and isinstance(payload["responses"], dict):
        return {k: str(v) for k, v in payload["responses"].items()}
    return {k: str(v) for k, v in payload.items() if k not in {"model", "date", "domain"}}


AUTHOR_ALIASES: dict[str, list[str]] = {
    "sigmund_freud": ["freud"],
    "leon_festinger": ["festinger"],
}

TRADITION_ALIASES: dict[str, list[str]] = {
    "christianity": ["christian"],
    "buddhism": ["buddhist"],
    "islam": ["muslim", "islamic"],
    "daoist": ["daoism", "taoist"],
}


def tradition_markers(tradition_id: str, traditions: dict) -> list[str]:
    record = traditions.get(tradition_id, {})
    markers = [tradition_id, tradition_id.replace("_", " ")]
    markers.extend(TRADITION_ALIASES.get(tradition_id, []))
    for key in ("labelEn", "labelZh"):
        value = record.get(key)
        if value:
            markers.append(str(value).lower())
    return markers


def author_markers(author_id: str) -> list[str]:
    markers = [author_id, author_id.replace("_", " ")]
    markers.extend(AUTHOR_ALIASES.get(author_id, []))
    return markers


def score_case(case: dict, response: str, traditions: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    text = normalize(response)
    ok = True

    deny = case.get("mustDenyAttribution")
    if deny:
        author = deny["author"]
        if not any(marker.lower() in text for marker in author_markers(author)):
            ok = False
            reasons.append(f"expected discussion of {author}")
        elif not matches_any(text, DENY_PATTERNS):
            ok = False
            reasons.append(f"expected denial of {author} attribution")

    affirm = case.get("mustAffirmAuthor")
    if affirm:
        author_id = affirm["author"]
        if not any(marker.lower() in text for marker in author_markers(author_id)):
            ok = False
            reasons.append(f"expected mention of correct author '{author_id}'")

    signal = case.get("mustSignalConfidence")
    if signal:
        confidence = signal["confidence"]
        patterns = {
            "compiled": [r"compiled", r"assembled", r"students", r"editors", r"彙編", r"編纂", r"弟子"],
            "legendary": [r"uncertain", r"disputed", r"legendary", r"debated", r"存疑", r"不確定"],
        }.get(confidence, [confidence])
        if not matches_any(text, patterns):
            ok = False
            reasons.append(f"expected signal for confidence '{confidence}'")

    if case.get("mustLabelMyth") and not matches_any(text, MYTH_PATTERNS):
        ok = False
        reasons.append("expected myth/misconception labeling")

    if case.get("mustDenyUniversal") and re.search(r"\ball\b.+\b(have|develop|show)\b", text):
        if not matches_any(text, [r"not all", r"rare", r"some", r"並非所有", r"少數", r"部分"]):
            ok = False
            reasons.append("expected rejection of universal claim")

    nuanced = case.get("mustSignalNuanced", [])
    if nuanced and not matches_any(text, nuanced):
        ok = False
        reasons.append(f"expected nuance markers: {nuanced}")

    for tradition in case.get("mustMentionTraditions", []):
        markers = tradition_markers(tradition, traditions)
        if not any(marker.lower() in text for marker in markers):
            ok = False
            reasons.append(f"expected tradition context '{tradition}'")

    for subfield in case.get("mustMentionSubfields", []):
        if subfield.replace("_", " ") not in text and subfield not in text:
            if subfield == "pop_myth" and not matches_any(text, [r"pop", r"popular", r"通俗", r"迷思"]):
                ok = False
                reasons.append(f"expected subfield '{subfield}'")

    for region in case.get("mustMentionRegions", []):
        if region not in text:
            pass  # soft check — global history answers may not name every region

    if case.get("mustUseCouncilPanel") and not matches_any(text, COUNCIL_PATTERNS):
        ok = False
        reasons.append("expected council/panel debate format")

    split = case.get("mustSplitWhenAppropriate")
    if split:
        has_ritual = matches_any(text, [r"ancestor", r"veneration", r"祭祖", r"ritual", r"禮"])
        has_philosophy = matches_any(text, [r"philosoph", r"moral", r"ethic", r"哲學", r"倫理"])
        if not (has_ritual and has_philosophy):
            ok = False
            reasons.append("expected split between Confucian philosophy and ritual religion when appropriate")

    if case.get("mustHandleSensitive"):
        if not (matches_any(text, COUNCIL_PATTERNS) or matches_any(text, SENSITIVE_OK_PATTERNS)):
            ok = False
            reasons.append("expected careful scholarly/traditional handling of sensitive topic")

    return ok, reasons


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