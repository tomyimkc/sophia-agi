# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Real HTTP transport for the crawler (production seam, stdlib-only).

The crawler (`pipeline.fetch.crawler`) is transport-injected; tests use a mock. This is the
*real* transport, built on stdlib ``urllib`` wrapped in ``asyncio.to_thread`` so it satisfies
the same ``async (url) -> (status, headers, body)`` contract without pulling in httpx/aiohttp.
It honors the environment's proxy (``HTTPS_PROXY``) and an optional CA bundle, sends a polite
User-Agent, caps body size, and surfaces HTTP status codes (4xx/5xx) as values so the crawler
can apply its retry/backoff policy. Network errors propagate so the crawler retries them.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import urllib.error
import urllib.request

_DEFAULT_CA = "/root/.ccr/ca-bundle.crt"


def make_http_transport(
    *,
    user_agent: str = "SophiaBot/0.1 (+https://github.com/tomyimkc/sophia-agi)",
    timeout: float = 20.0,
    max_bytes: int = 5_000_000,
    ca_bundle: str | None = None,
):
    """Return an async ``transport(url) -> (status, headers, body)`` backed by urllib.

    ``status`` is the HTTP status (200, 404, 503, ...). HTTP errors are returned as values;
    connection/timeout errors raise so the crawler's retry/backoff handles them.
    """
    ctx = ssl.create_default_context()
    ca = ca_bundle or os.environ.get("SSL_CERT_FILE") or _DEFAULT_CA
    if ca and os.path.isfile(ca):
        try:
            ctx.load_verify_locations(ca)
        except Exception:
            pass

    def _blocking(url):
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                raw = resp.read(max_bytes)
                headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, headers, raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # 4xx/5xx -> return the status as a value (crawler decides retry vs skip).
            body = ""
            try:
                body = e.read(max_bytes).decode("utf-8", errors="replace")
            except Exception:
                pass
            return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, body
        # URLError / timeout / socket errors propagate -> crawler retries with backoff.

    async def _transport(url):
        return await asyncio.to_thread(_blocking, url)

    return _transport


__all__ = ["make_http_transport"]
