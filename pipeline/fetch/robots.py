# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Polite robots.txt gating (Phase 4).

A well-behaved crawler honors robots.txt. ``RobotsCache`` fetches and caches each host's
robots rules through the **same injectable async transport** the crawler uses (so it needs no
network of its own and is fully testable), and answers ``await allowed(url)``. Parsing is
delegated to the stdlib ``urllib.robotparser``.

Convention: a *missing* robots.txt (404/empty) means "no rules" → allowed. A transport
*error* fetching robots is treated conservatively as disallowed for that host until it can be
read, so a flaky host isn't hammered. The transport returns ``(status, headers, body)``.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser


class RobotsCache:
    """Per-host robots.txt cache over an async ``transport(url) -> (status, headers, body)``."""

    def __init__(self, transport, *, user_agent: str = "SophiaBot"):
        self._transport = transport
        self._ua = user_agent
        self._cache: dict[str, RobotFileParser | None] = {}

    def _robots_url(self, url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme or "https", parts.netloc, "/robots.txt", "", ""))

    def _host_key(self, url: str) -> str:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}"

    async def _load(self, url: str) -> RobotFileParser | None:
        key = self._host_key(url)
        if key in self._cache:
            return self._cache[key]
        rp = RobotFileParser()
        try:
            status, _headers, body = await self._transport(self._robots_url(url))
        except Exception:
            # Robots fetch errored -> conservative disallow, but do NOT cache: a transient
            # failure must not block the host forever; the next call retries the fetch.
            return None
        if status == 200 and body:
            rp.parse(body.splitlines())
        else:
            rp.parse([])  # no robots.txt -> no restrictions
        self._cache[key] = rp
        return rp

    async def allowed(self, url: str) -> bool:
        """True iff ``user_agent`` may fetch ``url`` per the host's robots.txt."""
        rp = await self._load(url)
        if rp is None:
            return False
        return rp.can_fetch(self._ua, url)


__all__ = ["RobotsCache"]
