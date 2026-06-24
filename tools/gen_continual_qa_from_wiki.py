#!/usr/bin/env python3
"""Generate a CPQA episode stream from the real wiki/ corpus.

Loads the live OKF pages under wiki/, keeps the ones that are actually grounded, splits
them into domain-ordered episodes, and emits recall / retention / unlearning / control
queries — a ~100-query benchmark grounded in the real corpus rather than synthetic
pages. Deterministic (sorted order, no randomness).

    python tools/gen_continual_qa_from_wiki.py            # writes episodes_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import build_graph, is_grounded  # noqa: E402
from okf.page import load_pages  # noqa: E402

DEFAULT_OUT = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"
NEVER_TAUGHT = ["confucius_wrote_the_iching", "freud_coined_dissonance", "laozi_authored_pasta"]


def _meta(page) -> dict:
    return {k: v for k, v in page.meta.items() if k in
            ("id", "pageType", "domain", "attributedAuthor", "authorConfidence",
             "beliefTier", "derivesFrom", "contradicts", "canonicalTitleEn", "aliases")}


def _question(page) -> str:
    title = page.meta.get("canonicalTitleEn") or page.id.replace("_", " ")
    return f"What does the corpus record about {title}?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default=str(ROOT / "wiki"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    pages = load_pages(args.wiki)
    graph = build_graph(pages)
    grounded = [p for p in sorted(pages, key=lambda p: p.id) if is_grounded(graph, p.id)]

    # Group grounded pages into episodes by domain (fallback: pageType), deterministic.
    buckets: dict[str, list] = {}
    for p in grounded:
        key = str(p.meta.get("domain") or p.page_type or "misc")
        buckets.setdefault(key, []).append(p)

    episodes: list[dict] = []
    learned_so_far: list[str] = []
    bucket_items = sorted(buckets.items())
    for ei, (domain, group) in enumerate(bucket_items):
        learn = [_meta(p) for p in group]
        queries = [{"id": f"recall_{p.id}", "target": p.id, "expect": "assert",
                    "type": "recall", "q": _question(p)} for p in group]
        # Retention: re-ask up to 5 facts from earlier episodes.
        for j, prior in enumerate(learned_so_far[:5]):
            queries.append({"id": f"retain_{ei}_{prior}", "target": prior, "expect": "assert",
                            "type": "retention", "q": f"Do we still hold the entry '{prior.replace('_', ' ')}'?"})
        episodes.append({"episode": f"ep{ei + 1}_{domain}", "learn": learn, "retract": [], "queries": queries})
        learned_so_far.extend(p.id for p in group)

    # Final episode: deliberate unlearning of two leaf primaries + control abstains.
    # Retract from the FIRST episode (t0) so the frozen baseline — which can neither learn
    # nor unlearn — keeps asserting them (a fabrication), showing the staleness failure.
    t0_ids = {p.id for p in bucket_items[0][1]} if bucket_items else set()
    leaves = [p.id for p in grounded
              if not p.meta.get("derivesFrom")
              and not any(p.id in (d.meta.get("derivesFrom") or []) for d in grounded)]
    retract = [rid for rid in leaves if rid in t0_ids][:2] or leaves[:2]
    final_queries = []
    for rid in retract:
        final_queries.append({"id": f"unlearn_{rid}", "target": rid, "expect": "abstain",
                              "type": "unlearning", "q": f"After retraction, do we still assert '{rid.replace('_', ' ')}'?"})
    for nid in NEVER_TAUGHT:
        final_queries.append({"id": f"control_{nid}", "target": nid, "expect": "abstain",
                              "type": "control", "q": f"Is there an entry for '{nid.replace('_', ' ')}'?"})
    episodes.append({"episode": "epF_revision", "learn": [], "retract": retract, "queries": final_queries})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    total_q = sum(len(e["queries"]) for e in episodes)
    print(json.dumps({"out": args.out, "episodes": len(episodes), "groundedPages": len(grounded),
                      "queries": total_q, "retracted": retract}, indent=2))


if __name__ == "__main__":
    main()
