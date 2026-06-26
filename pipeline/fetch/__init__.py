# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Acquisition loop (Phase 4): frontier -> fetch -> extract -> feedback.

The JD's "全网数据采集环路" — a high-throughput, fault-tolerant collection loop whose URL
selection is driven by cleaning/quality feedback (Phase 1's ``link_priority``). The pieces:

  - ``pipeline.fetch.frontier.Frontier`` — a priority queue of URLs (canonicalized + deduped),
    seeded and continuously re-ranked from quality feedback;
  - ``pipeline.fetch.robots.RobotsCache`` — polite robots.txt gating;
  - ``pipeline.fetch.crawler.Crawler`` — async fetch with per-host rate limits, fetch quotas,
    and retry/backoff, over an **injectable transport** (tests + airgap use a mock; no network
    is required by this package);
  - ``pipeline.fetch.warc.iter_warc_records`` — ingest CommonCrawl-style WARC dumps so the loop
    can run at scale against an archived crawl without hitting the live web.

Everything here is stdlib-only and deterministic given a transport.
"""

from __future__ import annotations

__all__ = ["frontier", "robots", "crawler", "warc"]
