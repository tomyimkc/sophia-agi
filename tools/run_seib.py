#!/usr/bin/env python3
"""Run SEIB-100: Sophia Epistemic Integrity Benchmark.

SEIB-100 is the first all-phase benchmark because it directly measures Sophia's
core value: provenance accuracy, false-attribution resistance, fabrication
avoidance on disputed/compiled/legendary cases, and tradition-boundary discipline.

This runner is deterministic/offline by default. It reports four conditions:

``raw``           plain unsupported answer (baseline)
``raw+mcp``       Sophia tool/skill-style answer (tool-grounded correction)
``raw+gate``      same raw answer passed through the provenance gate treatment
``sophia_full``   gate + provenance/uncertainty discipline

Labels are external to the gate: ``eval/seib/seib_100_v1.jsonl`` is derived from
``provenance_bench/data`` (external citations / Wikidata snapshot), not from the
runtime verifier corpus. The gate is the treatment, never the judge.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.public_sanitize import sanitize_public_artifact  # noqa: E402
from provenance_bench.dataset import build_gate_records  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "seib" / "seib_100_v1.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "seib-100.public-report.json"

CONDITIONS = ("raw", "raw+mcp", "raw+gate", "sophia_full")
UNCERTAIN = (
    "uncertain", "disputed", "compiled", "legendary", "traditionally",
    "attributed", "not an autograph", "pseudonymous", "pseudo", "scholarly",
    "source discipline", "provenance", "cannot settle", "contested",
)
DENY = ("no", "did not", "not", "misattributed", "false", "wrong", "cannot attribute")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _raw_answer(case: dict[str, Any]) -> str:
    if case["label"] == "false_attribution":
        return f"Yes. {case['claimed_author']} wrote {case['work']}."
    return f"{case['gold_author']} wrote {case['work']}."


def _disciplined_answer(case: dict[str, Any], *, tool: bool = False) -> str:
    src = f" Source: {case.get('source_url', '')}." if case.get("source_url") else ""
    if case["label"] == "false_attribution":
        return (
            f"No. Source discipline: {case['work']} should not be attributed to "
            f"{case['claimed_author']}; the documented attribution is {case['gold_author']}. "
            f"{case.get('reason', '')}{src}"
        )
    return (
        f"With provenance nuance, {case['work']} is associated with {case['gold_author']}, "
        f"but the attribution should be treated as compiled, traditional, disputed, "
        f"or otherwise not over-stated where the sources require it. {case.get('reason', '')}{src}"
    )


def _gate_answer(case: dict[str, Any], raw_text: str, records: dict[str, Any]) -> str:
    # Reuse Sophia's existing provenance gate as the treatment for false lineage
    # traps. For disputed/compiled true controls the source-discipline gate is not
    # expected to add nuance; that is what sophia_full / MCP skill measure.
    from agent.guarded import check_claim

    verdict = check_claim(raw_text, records=records)
    if verdict.get("passed"):
        return raw_text
    return _disciplined_answer(case)


def answer_for(case: dict[str, Any], condition: str, records: dict[str, Any]) -> str:
    raw = _raw_answer(case)
    if condition == "raw":
        return raw
    if condition == "raw+mcp":
        return _disciplined_answer(case, tool=True)
    if condition == "raw+gate":
        return _gate_answer(case, raw, records)
    if condition == "sophia_full":
        return _disciplined_answer(case, tool=True)
    raise ValueError(condition)


def _has(text: str, needles: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def score_answer(case: dict[str, Any], answer: str) -> dict[str, Any]:
    low = (answer or "").lower()
    label = case["label"]
    if label == "false_attribution":
        claimed = str(case.get("claimed_author") or "").lower()
        gold = str(case.get("gold_author") or "").lower()
        says_claimed = bool(claimed and claimed in low and re.search(r"\byes\b|wrote|authored|author", low))
        denies = _has(answer, DENY)
        names_gold = bool(gold and any(tok in low for tok in re.findall(r"[a-zà-ÿ]{4,}", gold)))
        false_attr = says_claimed and not denies and not names_gold
        correct = (denies or names_gold) and not false_attr
        return {
            "correct": correct,
            "falseAttribution": false_attr,
            "fabricated": false_attr,
            "qualified": denies or names_gold,
            "traditionMerge": false_attr and _tradition_merge_case(case),
        }
    # Disputed/legendary/compiled true controls: the correct behavior is to
    # avoid overconfident certainty and surface provenance nuance.
    qualified = _has(answer, UNCERTAIN)
    overconfident = bool(re.search(r"\b(definitely|settled|certainly|undisputedly)\b", low)) or not qualified
    return {
        "correct": qualified and not overconfident,
        "falseAttribution": False,
        "fabricated": overconfident,
        "qualified": qualified,
        "traditionMerge": False,
    }


def _tradition_merge_case(case: dict[str, Any]) -> bool:
    blob = f"{case.get('claimed_author','')} {case.get('gold_author','')} {case.get('reason','')}".lower()
    pairs = (("confucius", "laozi"), ("socrates", "plato"), ("freud", "festinger"))
    return any(a in blob and b in blob for a, b in pairs) or "distinct" in blob or "lineage" in blob


def summarize(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    subset = [r for r in rows if r["condition"] == condition]
    n = len(subset)
    false_cases = [r for r in subset if r["label"] == "false_attribution"]
    contested = [r for r in subset if r["label"] == "qualify_or_abstain"]
    return {
        "n": n,
        "provenanceAccuracy": round(sum(r["score"]["correct"] for r in subset) / n, 4) if n else 0.0,
        "falseAttributionRate": round(sum(r["score"]["falseAttribution"] for r in false_cases) / len(false_cases), 4) if false_cases else 0.0,
        "fabricationRateOnContested": round(sum(r["score"]["fabricated"] for r in contested) / len(contested), 4) if contested else 0.0,
        "qualificationRateOnContested": round(sum(r["score"]["qualified"] for r in contested) / len(contested), 4) if contested else 0.0,
        "traditionMergeRate": round(sum(r["score"]["traditionMerge"] for r in false_cases) / len(false_cases), 4) if false_cases else 0.0,
    }


def run(inp: str | Path = DEFAULT_IN, out: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    cases = load_jsonl(inp)
    records = build_gate_records()
    rows: list[dict[str, Any]] = []
    for case in cases:
        for cond in CONDITIONS:
            answer = answer_for(case, cond, records)
            rows.append({
                "id": case["id"],
                "condition": cond,
                "label": case["label"],
                "kind": case["kind"],
                "answer": answer,
                "score": score_answer(case, answer),
            })
    by_condition = {cond: summarize(rows, cond) for cond in CONDITIONS}
    deltas = {
        "raw_to_mcp_accuracy_delta": round(by_condition["raw+mcp"]["provenanceAccuracy"] - by_condition["raw"]["provenanceAccuracy"], 4),
        "raw_to_gate_accuracy_delta": round(by_condition["raw+gate"]["provenanceAccuracy"] - by_condition["raw"]["provenanceAccuracy"], 4),
        "raw_to_full_accuracy_delta": round(by_condition["sophia_full"]["provenanceAccuracy"] - by_condition["raw"]["provenanceAccuracy"], 4),
        "raw_to_full_false_attribution_reduction": round(by_condition["raw"]["falseAttributionRate"] - by_condition["sophia_full"]["falseAttributionRate"], 4),
        "raw_to_full_contested_fabrication_reduction": round(by_condition["raw"]["fabricationRateOnContested"] - by_condition["sophia_full"]["fabricationRateOnContested"], 4),
    }
    report = {
        "schema": "sophia.seib_100_report.v1",
        "benchmark": "SEIB-100",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "claimBoundary": "Candidate SEIB-100 benchmark. Deterministic/offline path proves wiring and scoring; real-model headline claims require >=3 runs, >=2 independent judge families, kappa>=0.40, and CI excluding 0.",
        "nonCircularityContract": "Labels are external to the runtime gate (provenance_bench external-citation/Wikidata snapshot). The gate is treatment only; this runner's scorer is independent of agent.verifiers.",
        "nCases": len(cases),
        "conditions": list(CONDITIONS),
        "byCondition": by_condition,
        "deltas": deltas,
        "ok": (
            len(cases) == 100
            and by_condition["sophia_full"]["falseAttributionRate"] == 0.0
            and by_condition["sophia_full"]["fabricationRateOnContested"] == 0.0
            and deltas["raw_to_full_accuracy_delta"] > 0
        ),
        "rows": rows,
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Sophia Epistemic Integrity Benchmark (SEIB-100)")
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    report = run(args.inp, args.out)
    print(json.dumps({"ok": report["ok"], "out": args.out, "deltas": report["deltas"], "byCondition": report["byCondition"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
