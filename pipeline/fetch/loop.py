# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The acquisition loop, assembled (Phase 4).

Ties the pieces into the JD's 采集环路: seed the frontier → fetch politely → extract text +
links → score quality (Phase 1) → re-rank the frontier from that quality (feedback) → repeat.
Quality feedback is what makes this *selection*, not a blind spider: links discovered on a
high-quality site are enqueued at high priority, so the crawl drifts toward good neighborhoods
and away from junk — while dedup/quality still gate what is kept.

``run_loop`` is async and transport-injected, so it runs end-to-end with a mock (tests/airgap)
or a real httpx transport, unchanged.
"""

from __future__ import annotations

from pipeline.fetch.crawler import Crawler
from pipeline.fetch.extract import extract_links, extract_text
from pipeline.fetch.frontier import Frontier
from pipeline.link_priority import registered_domain
from pipeline.quality_score import score_document


async def run_loop(
    seeds,
    transport,
    *,
    robots=None,
    max_pages: int = 100,
    per_host_quota: int | None = None,
    follow_links: bool = True,
    seed_priority: float = 1.0,
    clock=None,
    sleep=None,
):
    """Run the crawl→score→feedback loop. Returns ``{"docs": [...], "stats": {...}}``.

    ``seeds`` is an iterable of URLs (or ``(url, priority)`` pairs). Each kept document carries a
    ``quality`` block; discovered links are re-queued at a priority equal to the source page's
    quality score (the feedback signal).
    """
    frontier = Frontier()
    for seed in seeds:
        if isinstance(seed, (tuple, list)):
            frontier.add(seed[0], seed[1])
        else:
            frontier.add(seed, seed_priority)

    crawler = Crawler(
        frontier,
        transport,
        robots=robots,
        per_host_quota=per_host_quota,
        max_pages=max_pages,
        clock=clock,
        sleep=sleep,
    )

    docs: list[dict] = []
    async for page in crawler.crawl():
        raw_html = page["content"]
        links = extract_links(page["url"], raw_html) if follow_links else []
        page["content"] = extract_text(raw_html)
        page["quality"] = score_document(page)
        docs.append(page)

        if follow_links:
            # Feedback: explore outlinks at a priority equal to this page's quality.
            frontier.feedback(links, page["quality"]["score"])

    # Per-site rollup for visibility (priority is recomputed from realized quality).
    by_domain: dict[str, list[float]] = {}
    for d in docs:
        by_domain.setdefault(registered_domain(d["url"]), []).append(d["quality"]["score"])

    stats = dict(crawler.stats)
    stats["kept"] = sum(1 for d in docs if d["quality"]["keep"])
    stats["sites"] = len([k for k in by_domain if k])
    return {"docs": docs, "stats": stats}


__all__ = ["run_loop"]
