# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""HTML link + text extraction (Phase 4), dependency-free.

Just enough extraction to drive the loop: pull ``<a href>`` links (resolved against the page
URL) so the frontier can be fed, and strip tags to plain text so the document can be scored.
Regex-based on purpose — no lxml/bs4 dependency, airgap-safe. Not a full HTML parser; good
enough for link discovery and a text proxy on well-formed pages.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

_HREF_RE = re.compile(r"""<a\b[^>]*?\bhref\s*=\s*["']([^"'#]+)["']""", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def extract_links(base_url: str, html: str) -> list[str]:
    """Absolute http(s) links found in ``<a href>``, resolved against ``base_url`` (deduped, ordered)."""
    out: list[str] = []
    seen: set[str] = set()
    for href in _HREF_RE.findall(html or ""):
        absolute = urljoin(base_url, href)
        if absolute.startswith(("http://", "https://")) and absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out


def extract_text(html: str) -> str:
    """Strip scripts/styles/tags to a whitespace-normalized text proxy."""
    no_scripts = _SCRIPT_STYLE_RE.sub(" ", html or "")
    text = _TAG_RE.sub(" ", no_scripts)
    return _WS_RE.sub(" ", text).strip()


__all__ = ["extract_links", "extract_text"]
