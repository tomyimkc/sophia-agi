#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Compounding-knowledge curve — does the wiki answer more as it grows? (AGI-proof)

Grows the OKF wiki in increments and, after each, measures how many held-out
provenance questions are answerable from the wiki via retrieval. A rising curve is
evidence of NON-PARAMETRIC growth: a frozen model answers more by reading a bigger
synthesized knowledge base, not by retraining — something a static LLM cannot show.

HONESTY: the default metric is `answerable-coverage@k` (a retrieval proxy), not a
semantic quality score. The model-backed answer-quality curve is the opt-in
extension (--provider) and requires the manual two-pass review noted in the
ablation caveats. A flat/declining curve is a real result and is reported as-is.

    python tools/run_compounding_curve.py        # offline coverage curve -> agi-proof/
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import page as okf_page  # noqa: E402

WIKI_DIR = ROOT / "wiki"
OUT_DIR = ROOT / "agi-proof" / "compounding-curve"
INCREMENTS = (0.2, 0.4, 0.6, 0.8, 1.0)


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9一-鿿]{3,}", text.lower())}


def _golden(pages: list) -> list:
    """Held-out provenance questions: (query, expected_page_id) for attributed pages."""
    golden = []
    for page in pages:
        title = page.meta.get("canonicalTitleEn") or page.meta.get("id")
        if page.meta.get("attributedAuthor") and page.meta.get("doNotAttributeTo") and title:
            golden.append((f"who wrote {title} attribution author", page.id))
    return golden


def _retrieve_topk(query: str, subset: list, *, k: int = 3) -> list:
    qt = _tokens(query)
    scored = []
    for page in subset:
        body_tokens = _tokens(f"{page.id} {page.meta.get('canonicalTitleEn') or ''} {page.body}")
        overlap = len(qt & body_tokens)
        if overlap:
            scored.append((overlap, page.id))
    scored.sort(reverse=True)
    return [pid for _, pid in scored[:k]]


def run(*, k: int = 3) -> dict:
    pages = sorted(okf_page.load_pages(WIKI_DIR), key=lambda p: p.id)
    golden = _golden(pages)
    total = len(golden)
    points = []
    for frac in INCREMENTS:
        n = max(1, int(round(frac * len(pages))))
        subset = pages[:n]
        subset_ids = {p.id for p in subset}
        hits = 0
        for query, expected in golden:
            if expected in subset_ids and expected in _retrieve_topk(query, subset, k=k):
                hits += 1
        coverage = round(hits / total, 4) if total else 0.0
        points.append({"wikiPages": len(subset), "answerableCoverage": coverage, "hits": hits, "ofGolden": total})
    slope = points[-1]["answerableCoverage"] - points[0]["answerableCoverage"] if len(points) > 1 else 0.0
    return {
        "metric": f"answerable-coverage@{k} (offline retrieval proxy; not a semantic quality score)",
        "goldenQuestions": total,
        "points": points,
        "slope": round(slope, 4),
        "rising": slope > 0,
        "note": "Model-backed answer-quality curve is the --provider extension; requires manual two-pass review.",
    }


def main() -> int:
    result = run()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "curve-latest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
