#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for pipeline.url_canonical (Phase 2).

Verifies tracking-param stripping, host/scheme normalization, www + default-port + trailing
slash handling, query stability (sorted), idempotence, and that param-variant URLs collapse
into one canonical cluster. Offline, no deps.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.url_canonical import canonical_clusters, canonicalize  # noqa: E402


def test_strips_tracking_params():
    assert canonicalize("https://x.com/a?utm_source=fb&utm_campaign=z&id=7") == "https://x.com/a?id=7"
    assert canonicalize("https://x.com/a?fbclid=abc&gclid=def") == "https://x.com/a"
    assert canonicalize("https://x.com/a?sessionid=xyz&ref=agg") == "https://x.com/a"


def test_normalizes_host_scheme_port_slash():
    assert canonicalize("HTTPS://WWW.X.COM:443/a/") == "https://x.com/a"
    assert canonicalize("http://x.com:80//a//b/") == "http://x.com/a/b"
    assert canonicalize("https://x.com/#frag") == "https://x.com/"


def test_query_sorted_and_stable():
    assert canonicalize("https://x.com/a?b=2&a=1") == "https://x.com/a?a=1&b=2"


def test_idempotent():
    u = "https://WWW.x.com/a/?utm_source=q&z=1&a=2#frag"
    once = canonicalize(u)
    assert canonicalize(once) == once


def test_non_url_passthrough():
    assert canonicalize("not a url") == "not a url"


def test_param_variants_collapse():
    urls = [
        "https://x.com/p?utm_source=a",
        "https://www.x.com/p?utm_source=b",
        "https://x.com/p/",
        "https://x.com/other",
    ]
    clusters = canonical_clusters(urls)
    # First three collapse to one canonical URL; the fourth is separate.
    assert len(clusters) == 2
    assert max(len(v) for v in clusters.values()) == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all pipeline.url_canonical tests passed")
