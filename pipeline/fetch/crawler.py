# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Async, polite, fault-tolerant crawler (Phase 4).

Drives the acquisition loop over an **injectable async transport**, so the politeness and
fault-tolerance logic is exercised in tests and airgap with a mock — no real network is
required by this package. Production use supplies an httpx/aiohttp-backed transport with the
same signature: ``async transport(url) -> (status:int, headers:dict, body:str)``.

Politeness / robustness (the JP's 高吞吐、可容错):
  - **robots.txt** gating (``RobotsCache``);
  - **per-host rate limit** — minimum interval between fetches to one host;
  - **per-host fetch quota** + global **max_pages** (抓取配额) — bounds crawl size and keeps
    diversity;
  - **retry with exponential backoff** on transport errors / 5xx / 429.

Time is injectable (``clock`` + async ``sleep``) so rate-limit and backoff behavior is
deterministic under test.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit


class Crawler:
    def __init__(
        self,
        frontier,
        transport,
        *,
        robots=None,
        user_agent: str = "SophiaBot",
        per_host_interval: float = 0.0,
        per_host_quota: int | None = None,
        max_pages: int | None = None,
        max_retries: int = 3,
        backoff_base: float = 0.2,
        clock=None,
        sleep=None,
    ):
        self.frontier = frontier
        self.transport = transport
        self.robots = robots
        self.user_agent = user_agent
        self.per_host_interval = per_host_interval
        self.per_host_quota = per_host_quota
        self.max_pages = max_pages
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._clock = clock or (lambda: 0.0)
        self._sleep = sleep or asyncio.sleep
        self._host_last: dict[str, float] = {}
        self._host_count: dict[str, int] = {}
        self.stats = {
            "fetched": 0,
            "skipped_robots": 0,
            "skipped_quota": 0,
            "errors": 0,
            "retries": 0,
        }

    @staticmethod
    def _host(url: str) -> str:
        return urlsplit(url).netloc

    async def _rate_limit(self, host: str) -> None:
        if self.per_host_interval <= 0:
            return
        last = self._host_last.get(host)
        now = self._clock()
        if last is not None:
            wait = self.per_host_interval - (now - last)
            if wait > 0:
                await self._sleep(wait)
        self._host_last[host] = self._clock()

    async def _fetch_with_retry(self, url: str):
        """Fetch with exponential backoff on transport error / 5xx / 429. Returns a 3-tuple or None."""
        for attempt in range(self.max_retries + 1):
            try:
                status, headers, body = await self.transport(url)
            except Exception:
                status = None
                headers, body = {}, ""
            if status is not None and status < 500 and status != 429:
                return status, headers, body
            if attempt < self.max_retries:
                self.stats["retries"] += 1
                await self._sleep(self.backoff_base * (2**attempt))
        return None

    async def crawl(self):
        """Async generator yielding fetched documents until the frontier drains or limits hit.

        Each yielded doc: ``{url, status, mime, content}``. The caller is expected to score the
        doc and call ``frontier.feedback(links, priority)`` to re-rank — that is the loop.
        """
        while len(self.frontier) and (self.max_pages is None or self.stats["fetched"] < self.max_pages):
            url = self.frontier.pop()
            if url is None:
                break
            host = self._host(url)

            if self.per_host_quota is not None and self._host_count.get(host, 0) >= self.per_host_quota:
                self.stats["skipped_quota"] += 1
                continue
            if self.robots is not None and not await self.robots.allowed(url):
                self.stats["skipped_robots"] += 1
                continue

            await self._rate_limit(host)
            result = await self._fetch_with_retry(url)
            if result is None or result[0] >= 400:
                self.stats["errors"] += 1
                continue

            status, headers, body = result
            self._host_count[host] = self._host_count.get(host, 0) + 1
            self.stats["fetched"] += 1
            yield {
                "url": url,
                "status": status,
                "mime": (headers or {}).get("content-type", "text/html"),
                "content": body or "",
            }


def dict_transport(pages: dict, *, robots: dict | None = None):
    """Build an async transport backed by in-memory ``{url: (status, headers, body)}``.

    For tests, airgap demos, and replaying a fixed page set. ``robots`` optionally maps a
    robots.txt URL to its body. Unknown URLs return 404.
    """
    table = dict(pages)
    for rurl, rbody in (robots or {}).items():
        table.setdefault(rurl, (200, {"content-type": "text/plain"}, rbody))

    async def _transport(url):
        if url in table:
            return table[url]
        if url.endswith("/robots.txt"):
            return 404, {}, ""
        return 404, {}, ""

    return _transport


__all__ = ["Crawler", "dict_transport"]
