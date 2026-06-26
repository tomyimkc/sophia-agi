#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Phase 4 acquisition loop (pipeline.fetch).

Covers: frontier priority + canonical dedup + feedback; robots gating (allow/deny/missing);
crawler per-host quota, rate limiting, and retry-with-backoff over a mock transport; HTML
link/text extraction; WARC response-record parsing; and the assembled crawl→score→feedback
loop. Fully offline — a dict-backed async transport stands in for the network. No deps.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.fetch.crawler import Crawler, dict_transport  # noqa: E402
from pipeline.fetch.extract import extract_links, extract_text  # noqa: E402
from pipeline.fetch.frontier import Frontier  # noqa: E402
from pipeline.fetch.loop import run_loop  # noqa: E402
from pipeline.fetch.robots import RobotsCache  # noqa: E402
from pipeline.fetch.warc import iter_warc_records, records_to_documents  # noqa: E402


# ------------------------------- frontier ---------------------------------- #

def test_frontier_priority_order():
    f = Frontier()
    f.add("https://a.com/low", 0.1)
    f.add("https://a.com/high", 0.9)
    f.add("https://a.com/mid", 0.5)
    assert [f.pop(), f.pop(), f.pop()] == [
        "https://a.com/high",
        "https://a.com/mid",
        "https://a.com/low",
    ]
    assert f.pop() is None


def test_frontier_dedup_by_canonical():
    f = Frontier()
    assert f.add("https://a.com/p?utm_source=x", 0.5) is True
    assert f.add("https://www.a.com/p", 0.9) is False  # same canonical -> not re-added
    f.pop()
    assert f.add("https://a.com/p", 0.9) is False  # already seen


def test_frontier_feedback():
    f = Frontier()
    f.add("https://a.com/seed", 1.0)
    f.pop()
    added = f.feedback(["https://a.com/x", "https://a.com/seed"], 0.7)
    assert added == 1  # seed already seen; only /x is new


# -------------------------------- robots ----------------------------------- #

def test_robots_allow_and_deny():
    robots_body = "User-agent: *\nDisallow: /private\n"
    transport = dict_transport({}, robots={"https://a.com/robots.txt": robots_body})
    rc = RobotsCache(transport, user_agent="SophiaBot")
    assert asyncio.run(rc.allowed("https://a.com/public")) is True
    assert asyncio.run(rc.allowed("https://a.com/private/x")) is False


def test_robots_missing_allows():
    rc = RobotsCache(dict_transport({}), user_agent="SophiaBot")
    assert asyncio.run(rc.allowed("https://nope.com/anything")) is True


# -------------------------------- crawler ---------------------------------- #

def _collect(crawler):
    async def go():
        return [d async for d in crawler.crawl()]

    return asyncio.run(go())


def test_crawler_basic_fetch():
    f = Frontier()
    f.add("https://a.com/1", 1.0)
    f.add("https://a.com/2", 0.9)
    transport = dict_transport(
        {
            "https://a.com/1": (200, {"content-type": "text/html"}, "<p>one</p>"),
            "https://a.com/2": (200, {"content-type": "text/html"}, "<p>two</p>"),
        }
    )
    docs = _collect(Crawler(f, transport))
    assert {d["url"] for d in docs} == {"https://a.com/1", "https://a.com/2"}


def test_crawler_per_host_quota():
    f = Frontier()
    for i in range(5):
        f.add(f"https://a.com/{i}", 1.0)
    transport = dict_transport({f"https://a.com/{i}": (200, {}, "x") for i in range(5)})
    crawler = Crawler(f, transport, per_host_quota=2)
    docs = _collect(crawler)
    assert len(docs) == 2
    assert crawler.stats["skipped_quota"] == 3


def test_crawler_retries_then_succeeds():
    calls = {"n": 0}

    async def flaky(url):
        calls["n"] += 1
        if calls["n"] < 3:
            return 503, {}, ""  # retryable
        return 200, {"content-type": "text/html"}, "ok"

    f = Frontier()
    f.add("https://a.com/x", 1.0)
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    crawler = Crawler(f, flaky, max_retries=3, sleep=fake_sleep)
    docs = _collect(crawler)
    assert len(docs) == 1 and docs[0]["content"] == "ok"
    assert crawler.stats["retries"] == 2
    assert slept == [0.2, 0.4]  # exponential backoff


def test_crawler_rate_limit_waits():
    t = {"now": 0.0}
    waited = []

    async def fake_sleep(s):
        waited.append(s)
        t["now"] += s

    f = Frontier()
    f.add("https://a.com/1", 1.0)
    f.add("https://a.com/2", 0.9)
    transport = dict_transport({"https://a.com/1": (200, {}, "a"), "https://a.com/2": (200, {}, "b")})
    crawler = Crawler(f, transport, per_host_interval=1.0, clock=lambda: t["now"], sleep=fake_sleep)
    _collect(crawler)
    assert any(w > 0 for w in waited)  # second same-host fetch waited


# -------------------------------- extract ---------------------------------- #

def test_extract_links_and_text():
    html = '<html><body>Hello <a href="/a">x</a> <a href="https://b.com/c">y</a>' \
           '<script>ignore()</script> world</body></html>'
    links = extract_links("https://a.com/page", html)
    assert "https://a.com/a" in links
    assert "https://b.com/c" in links
    text = extract_text(html)
    assert "Hello" in text and "world" in text and "ignore" not in text


# --------------------------------- WARC ------------------------------------ #

_WARC = (
    b"WARC/1.0\r\n"
    b"WARC-Type: response\r\n"
    b"WARC-Target-URI: https://a.com/page\r\n"
    b"Content-Length: 78\r\n"
    b"\r\n"
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/html\r\n"
    b"\r\n"
    b"<html><body>hello warc world</body></html>\r\n"
    b"\r\n"
    b"WARC/1.0\r\n"
    b"WARC-Type: request\r\n"
    b"WARC-Target-URI: https://a.com/page\r\n"
    b"Content-Length: 5\r\n"
    b"\r\n"
    b"GET /\r\n"
    b"\r\n"
)


def test_warc_parsing():
    records = list(iter_warc_records(_WARC))
    types = {r["warc_type"] for r in records}
    assert "response" in types and "request" in types
    docs = list(records_to_documents(records))
    assert len(docs) == 1
    assert docs[0]["url"] == "https://a.com/page"
    assert "hello warc world" in docs[0]["content"]


# ------------------------------ full loop ---------------------------------- #

def test_run_loop_crawls_scores_and_follows():
    pages = {
        "https://good.com/index": (
            200,
            {"content-type": "text/html"},
            '<html><body>A long, clean paragraph of ordinary encyclopedic prose about a '
            'reasonable topic with plenty of distinct vocabulary, linking onward. '
            '<a href="/more">more</a></body></html>',
        ),
        "https://good.com/more": (
            200,
            {"content-type": "text/html"},
            "<html><body>Another substantive page of clean prose continuing the article "
            "with further detail and distinct words to clear the length threshold.</body></html>",
        ),
    }
    result = asyncio.run(
        run_loop(["https://good.com/index"], dict_transport(pages), max_pages=10)
    )
    urls = {d["url"] for d in result["docs"]}
    # Followed the discovered link from the seed page.
    assert "https://good.com/index" in urls
    assert "https://good.com/more" in urls
    assert result["stats"]["fetched"] == 2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.fetch tests passed")
