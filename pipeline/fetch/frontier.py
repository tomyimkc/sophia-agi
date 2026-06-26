# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""URL frontier with quality-feedback re-ranking (Phase 4).

The frontier is the heart of the acquisition loop's "data selection": a priority queue of
URLs to fetch, where priority comes from cleaning/quality feedback (Phase 1). High-value
sites surface first; already-seen URLs (after canonicalization) are never re-queued, which
also enforces the JD's 链接规模控制 at the queue level.

Deterministic: ties break by insertion order, so a fixed seed + feedback sequence always
produces the same crawl order. Stdlib-only (``heapq``).
"""

from __future__ import annotations

import heapq

from pipeline.url_canonical import canonicalize


class Frontier:
    """A max-priority queue of URLs, deduplicated by canonical form.

    ``add(url, priority)`` enqueues unseen URLs; ``pop()`` returns the highest-priority URL.
    Priority is any float (e.g. a site's ``link_priority`` score); higher fetches first.
    """

    def __init__(self):
        self._heap: list[tuple[float, int, str]] = []  # (-priority, seq, canonical_url)
        self._seen: set[str] = set()
        self._queued: set[str] = set()
        self._seq = 0

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, url: str, priority: float = 0.0) -> bool:
        """Enqueue ``url`` if not already seen/queued. Returns True if it was added."""
        canon = canonicalize(url)
        if not canon or canon in self._seen or canon in self._queued:
            return False
        heapq.heappush(self._heap, (-float(priority), self._seq, canon))
        self._queued.add(canon)
        self._seq += 1
        return True

    def add_many(self, urls, priority: float = 0.0) -> int:
        """Enqueue several URLs at one priority; returns how many were newly added."""
        return sum(1 for u in urls if self.add(u, priority))

    def pop(self) -> str | None:
        """Return the highest-priority URL (marking it seen), or None if empty."""
        while self._heap:
            _neg, _seq, canon = heapq.heappop(self._heap)
            self._queued.discard(canon)
            if canon in self._seen:
                continue
            self._seen.add(canon)
            return canon
        return None

    def feedback(self, links, priority: float) -> int:
        """Re-rank: enqueue newly-discovered ``links`` at a feedback ``priority``.

        This closes the loop — after a fetched page is scored, its outlinks are queued at a
        priority derived from the page's site quality, so good neighborhoods are explored first.
        """
        return self.add_many(links, priority)

    @property
    def seen_count(self) -> int:
        return len(self._seen)


__all__ = ["Frontier"]
