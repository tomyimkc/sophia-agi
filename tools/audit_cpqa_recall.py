#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Step 6 — audit CPQA recall failures by source sufficiency (offline, deterministic).

The full-92 cross-gateway run showed grounded recall at 0.50 vs raw 0.94. Before changing
the gate we measure *why*: for every recall (assert) query, is the retrieved OKF/wiki page
actually answer-bearing, or a thin provenance stub? A grounded system cannot answer from a
stub no matter how good the model — so the share of thin-source queries is the ceiling on
how much any prompt/gate change can recover, and the rest needs richer sources or
graph-neighborhood retrieval.

Heuristic (deterministic, no LLM): a page is *answer-bearing* if its Markdown body carries
at least one free-prose line (a real sentence, not a `#`/`-`/`>`/`_`/`|` template line) of
reasonable length, beyond the generated provenance scaffold. Otherwise it is *thin*.

    python tools/audit_cpqa_recall.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa import load_episodes  # noqa: E402
from okf.page import load_pages  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "continual-qa.recall-audit.json"
_TEMPLATE_PREFIXES = ("#", "-", ">", "_", "|", "*")


def freeform_prose_lines(body: str, *, min_len: int = 40) -> "list[str]":
    """Body lines that read as real prose, not the generated provenance scaffold."""
    out = []
    for raw in (body or "").splitlines():
        line = raw.strip()
        if len(line) < min_len:
            continue
        if line.startswith(_TEMPLATE_PREFIXES):
            continue
        out.append(line)
    return out


def classify_source(page) -> "dict":
    body = page.body or ""
    prose = freeform_prose_lines(body)
    prose_chars = sum(len(p) for p in prose)
    return {
        "bodyChars": len(body.strip()),
        "proseLines": len(prose),
        "proseChars": prose_chars,
        # Answer-bearing only if there is real prose beyond the scaffold.
        "answerBearing": bool(prose) and prose_chars >= 60,
    }


def audit(episodes_path, wiki_path) -> "dict":
    episodes = load_episodes(episodes_path)
    pages = {p.id: p for p in load_pages(wiki_path)}

    recall_targets: list[str] = []
    for ep in episodes:
        for q in ep.queries:
            if q.expect == "assert":
                recall_targets.append(q.target)
    recall_targets = sorted(set(recall_targets))

    rows = []
    thin = []
    for tid in recall_targets:
        page = pages.get(tid)
        if page is None:
            rows.append({"target": tid, "found": False, "answerBearing": False})
            thin.append(tid)
            continue
        c = classify_source(page)
        rows.append({"target": tid, "found": True, **c})
        if not c["answerBearing"]:
            thin.append(tid)

    n = len(recall_targets)
    thin_n = len(thin)
    return {
        "schema": "sophia.cpqa_recall_audit.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "recallTargets": n,
        "thinSource": thin_n,
        "answerBearing": n - thin_n,
        "thinSourceShare": round(thin_n / n, 4) if n else 0.0,
        "interpretation": (
            "thinSourceShare is the fraction of recall queries whose retrieved page is a "
            "provenance stub with no answer-bearing prose. Grounded recall is capped at "
            f"~{round(1 - thin_n / n, 4) if n else 0.0} by the corpus alone; the rest needs "
            "richer source bodies (Step 5) or graph-neighborhood retrieval (Step 1)."
        ),
        "thinTargets": thin,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=str(DEFAULT_IN))
    ap.add_argument("--wiki", default=str(ROOT / "wiki"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    report = audit(args.episodes, args.wiki)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("recallTargets", "thinSource", "answerBearing", "thinSourceShare")}, indent=2))
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
