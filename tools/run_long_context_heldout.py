#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Held-out, NON-synthetic long-context recall eval for Sophia.

Unlike the synthetic harness (templated filler + LC_NEEDLE_* tokens), every item
here is real-world prose stating a real fact (the needle), embedded among real,
plausible distractor documents (genuine competing people/dates). Recall is scored
by deterministic exact-match against the gold answer tokens — no LLM judge.

Honest scope (CANDIDATE, never validated):
  - Documents state well-established public facts; the questions/labels are
    maintainer-authored (not model-authored), so this is a held-out probe, not a
    third-party benchmark. Small N.
  - Under the offline MOCK backend a model cannot read prose, so recall is ~0 and
    the run is a PLUMBING + corpus-integrity check only. Real recall requires a
    real model: --backend adapter (SOPHIA_MODEL_PROVIDER=ollama|mlx).
  - Relevance defaults to the offline lexical-vector retriever (real docs carry no
    hand-set scores), so this exercises real retrieval-under-distractors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_hidden_eval_sophia import RunConfig, backend_preflight, run_case  # noqa: E402
from tools.run_long_context_sophia import (  # noqa: E402
    GATED_PACKED_MODE,
    RAW_BASELINE_MODE,
    average,
    ci95,
    long_context_modes,
    validate_context_pack_card,
)

NEEDLES = ROOT / "agi-proof" / "long-context" / "heldout" / "heldout-needles.jsonl"
REPORT_DIR = ROOT / "agi-proof" / "long-context" / "heldout"


def load_items(path: Path = NEEDLES) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def corpus_integrity(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Sharp-test guarantees: every gold answer token is exact-match present ONLY in its
    gold doc (so the answer can't be lifted from a distractor), and every distractor token
    is present in a distractor doc (so the distractors are real, not strawmen)."""
    failures: list[str] = []
    for item in items:
        gold_text = item["goldDoc"]["text"]
        distractor_text = " ".join(d["text"] for d in item["distractorDocs"])
        for token in item["answerTokens"]:
            if token not in gold_text:
                failures.append(f"{item['id']}: gold token {token!r} absent from gold doc")
            if token in distractor_text:
                failures.append(f"{item['id']}: gold token {token!r} leaks into a distractor doc")
        for token in item["distractorTokens"]:
            if token not in distractor_text:
                failures.append(f"{item['id']}: distractor token {token!r} absent from distractor docs")
    return {"clean": not failures, "failures": failures}


def build_case(item: dict[str, Any], *, budget_tokens: int, relevance_source: str) -> dict[str, Any]:
    # Order: distractor, gold (middle), distractor — gold is not first, so trivial
    # head-of-context truncation does not hand over the answer.
    distractors = item["distractorDocs"]
    ordered = []
    if distractors:
        ordered.append(distractors[0])
    ordered.append({**item["goldDoc"], "_gold": True})
    ordered.extend(distractors[1:])

    passages = []
    for idx, doc in enumerate(ordered):
        is_gold = bool(doc.get("_gold"))
        passages.append(
            {
                "id": f"{item['id']}_p{idx}",
                "sourceId": f"heldout://{item['id']}/{doc['id']}",
                "depthPct": round(idx / max(len(ordered) - 1, 1) * 100),
                "text": doc["text"],
                "answerTokens": list(item["answerTokens"]) if is_gold else [],
            }
        )
    return {
        "id": item["id"],
        "domain": "factual",
        "prompt": item["question"] + " Answer only from the provided documents.",
        "materials": [],
        "longContextPassages": passages,
        "relevanceSource": relevance_source,
        "contextBudgetTokens": budget_tokens,
        "answerTokens": list(item["answerTokens"]),
        "distractorTokens": list(item["distractorTokens"]),
        "scoring": {"mustInclude": item["answerTokens"], "mustAvoid": item["distractorTokens"]},
    }


def score_item(case: dict[str, Any], mode_name: str, ablation: Any, config: RunConfig) -> dict[str, Any]:
    started = time.time()
    result = run_case(case, f"heldout-{mode_name}", config=config, ablation=ablation)
    answer = result["answer"]
    recalled = [t for t in case["answerTokens"] if t in answer]
    wrong = [t for t in case["distractorTokens"] if t in answer]
    card = result.get("contextPackCard", {})
    return {
        "caseId": case["id"],
        "mode": mode_name,
        "recall": round(len(recalled) / len(case["answerTokens"]), 6) if case["answerTokens"] else 0.0,
        "wrongDistractorTokenCount": len(wrong),
        "answerSpanPresentInPack": bool(card.get("answer_span_present_in_pack")),
        "cardErrors": validate_context_pack_card(card) if card else ["no card"],
        "wallTimeSec": round(time.time() - started, 6),
    }


def build_report(items, rows, *, backend, relevance_source, integrity, budget_tokens) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    raw = [r["recall"] for r in by_mode.get(RAW_BASELINE_MODE, [])]
    gated = [r["recall"] for r in by_mode.get(GATED_PACKED_MODE, [])]
    paired = [g - r for g, r in zip(gated, raw)]
    is_mock = backend == "mock"
    return {
        "schema": "sophia.long_context_heldout.public_report.v1",
        "reportStatus": "candidate",
        "claimStatus": "candidate_not_validated",
        "candidateOnly": True,
        "canClaimAGI": False,
        "backend": backend,
        "relevanceSource": relevance_source,
        "budgetTokens": budget_tokens,
        "items": len(items),
        "corpusIntegrity": integrity,
        "claimBoundary": (
            "CANDIDATE: real-fact prose with real distractors, maintainer-authored questions, "
            "deterministic exact-match scoring, small N, not a third-party benchmark."
        ),
        "backendClaimBoundary": (
            "MOCK backend cannot read prose; recall ~0 here is a PLUMBING + corpus-integrity check only. "
            "Run --backend adapter (SOPHIA_MODEL_PROVIDER=ollama|mlx) for real recall."
            if is_mock
            else f"Recall from a single local model backend ({backend}); candidate, no external replication."
        ),
        "recallByMode": {mode: average([r["recall"] for r in rs]) for mode, rs in sorted(by_mode.items())},
        "headlineMetric": {
            "metric": "gated_packed_recall - raw_truncated_recall (real-fact held-out)",
            "rawTruncatedRecall": average(raw),
            "gatedPackedRecall": average(gated),
            "delta": average(paired),
            "gridDispersion95": ci95(paired),
            "pairedItems": len(paired),
        },
        "wrongDistractorTokenRateByMode": {
            mode: average([1.0 if r["wrongDistractorTokenCount"] else 0.0 for r in rs])
            for mode, rs in sorted(by_mode.items())
        },
        "cardsValid": all(not r["cardErrors"] for r in rows),
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "notes": [
            "Non-synthetic: real-world facts in real prose with real competing distractors.",
            "Exact-match scoring against gold tokens; no LLM judge.",
            "Maintainer-authored questions/labels; small N; candidate, not third-party validated.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out non-synthetic long-context recall eval (offline)")
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--relevance", choices=["synthetic", "lexical"], default="lexical")
    parser.add_argument("--budget-tokens", type=int, default=140)
    parser.add_argument("--timeout-sec", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.backend != "mock":
        preflight = backend_preflight(backend=args.backend, timeout_sec=max(args.timeout_sec, 30))
        if not preflight.get("ok"):
            print(json.dumps({"ok": False, "stage": "backend-preflight", "backend": args.backend,
                              "localModelUnavailable": True, "note": "no reachable local model; no report written"}, indent=2))
            return 0

    items = load_items()
    integrity = corpus_integrity(items)
    if not integrity["clean"]:
        print(json.dumps({"ok": False, "stage": "corpus-integrity", "failures": integrity["failures"]}, indent=2))
        return 1

    modes = long_context_modes(f"{RAW_BASELINE_MODE},{GATED_PACKED_MODE}")
    config = RunConfig(backend=args.backend, timeout_sec=args.timeout_sec)
    rows: list[dict[str, Any]] = []
    for item in items:
        case = build_case(item, budget_tokens=args.budget_tokens, relevance_source=args.relevance)
        for mode_name, ablation in modes.items():
            rows.append(score_item(case, mode_name, ablation, config))

    report = build_report(items, rows, backend=args.backend, relevance_source=args.relevance,
                          integrity=integrity, budget_tokens=args.budget_tokens)
    if not report["cardsValid"]:
        print(json.dumps({"ok": False, "stage": "card-validation",
                          "errors": [r["cardErrors"] for r in rows if r["cardErrors"]]}, indent=2))
        return 1

    out = args.out or (REPORT_DIR / "heldout-candidate.public-report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    print(json.dumps({k: report[k] for k in ("reportStatus", "backend", "relevanceSource", "recallByMode", "corpusIntegrity")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
