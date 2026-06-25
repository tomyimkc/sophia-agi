# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""URL canonicalization and link-scale control (Phase 2).

The JD's "链接规模控制 ... 清洗 URL 冗余参数、规范化链接" — collapse the many URL
spellings of the same resource so the link library stays small and clean. This module is
the deterministic, stdlib-only normalizer that runs before dedup:

  - lowercase scheme + host, drop default ports and a leading ``www.``;
  - drop the fragment and known tracking/session params (utm_*, fbclid, gclid, ref,
    sessionid, ...), keeping meaningful query params sorted for stability;
  - collapse duplicate slashes and a trailing slash in the path.

Same-host param variants collapse here; cross-host *mirror* sites (same content, different
domain) are caught downstream by content dedup (``pipeline.dedup.minhash``).
"""

from __future__ import annotations

import re
from collections import defaultdict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

#: Query parameters that never identify a distinct resource (analytics / session / referrer).
_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "ref",
        "referrer",
        "source",
        "sessionid",
        "session_id",
        "sid",
        "spm",
        "yclid",
        "_ga",
        "campaign",
    }
)
#: Param *prefixes* to drop (utm_source, utm_medium, ...).
_TRACKING_PREFIXES = ("utm_",)
_DEFAULT_PORTS = {"http": "80", "https": "443"}
_MULTI_SLASH_RE = re.compile(r"/{2,}")


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k in _TRACKING_PARAMS or k.startswith(_TRACKING_PREFIXES)


def canonicalize(url: str) -> str:
    """Return a canonical form of ``url`` (best-effort; returns input unchanged on parse error).

    Idempotent: ``canonicalize(canonicalize(u)) == canonicalize(u)``.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    if not parts.scheme and not parts.netloc:
        return url

    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host
    if parts.port and str(parts.port) != _DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    path = _MULTI_SLASH_RE.sub("/", parts.path)
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)]
    kept.sort()
    query = urlencode(kept)

    return urlunsplit((scheme, netloc, path, query, ""))


def canonical_clusters(urls) -> dict[str, list[str]]:
    """Group URLs by their canonical form: ``{canonical_url: [original, ...]}``."""
    clusters: dict[str, list[str]] = defaultdict(list)
    for u in urls:
        clusters[canonicalize(u)].append(u)
    return dict(clusters)


def annotate(doc: dict) -> dict:
    """Set ``doc['canonical_url']`` from ``doc['url']`` in place; return the doc."""
    if doc.get("url"):
        doc["canonical_url"] = canonicalize(doc["url"])
    return doc


__all__ = ["canonicalize", "canonical_clusters", "annotate"]
