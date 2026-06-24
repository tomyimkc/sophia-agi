# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shared types for legal-authority sources (HKLII, e-Legislation, ...).

A ``LegalSource`` answers one narrow, machine-checkable question: *does this
citation resolve to a real authority in a trusted primary source?* Every source
is **fail-closed** — a network error, timeout, or ambiguous match yields
``verified=False`` with a status, never a silent pass.

No third-party deps: HTTP is plain ``urllib`` (mirrors ``agent/web_evidence.py``),
and the ``fetch`` callable is injectable so tests never touch the network.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol

# (status_code, body_text). Raisers should be wrapped by the caller into a
# fail-closed Resolution — a source must never propagate a network exception.
Fetch = Callable[[str, int], "tuple[int, str]"]

USER_AGENT = "sophia-agi-legal-verifier/0.1 (+https://github.com/tomyimkc/sophia-agi)"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class Resolution:
    """The outcome of trying to verify one citation against a primary source."""

    citation: str
    verified: bool
    status: str  # verified | not_found | error | offline | unsupported
    provider: str  # hklii | elegislation | cache | mock | none
    sourceType: str = ""  # case | legislation
    title: str = ""
    court: str = ""
    year: str = ""
    url: str = ""
    retrievedAt: str = ""
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in ("", [], None)}


def verified(citation: str, provider: str, *, source_type: str, url: str = "", **meta) -> Resolution:
    return Resolution(
        citation=citation, verified=True, status="verified", provider=provider,
        sourceType=source_type, url=url, retrievedAt=now_iso(),
        title=meta.get("title", ""), court=meta.get("court", ""), year=meta.get("year", ""),
    )


def unverified(citation: str, provider: str, status: str, reason: str, *, source_type: str = "") -> Resolution:
    """Fail-closed result. ``status`` is one of not_found/error/offline/unsupported."""
    return Resolution(
        citation=citation, verified=False, status=status, provider=provider,
        sourceType=source_type, retrievedAt=now_iso(), reasons=[reason],
    )


class LegalSource(Protocol):
    """A primary-source backend for one family of citations."""

    name: str

    def can_resolve(self, citation: str) -> bool:
        """True if this source handles the citation's shape/jurisdiction."""

    def resolve(self, citation: str, *, fetch: Fetch, timeout: int = 20) -> Resolution:
        """Verify the citation. MUST be fail-closed (never raise)."""


def loose_contains(body: str, citation: str) -> bool:
    """True if ``citation`` appears in ``body`` allowing flexible whitespace.

    Works for both HTML (HKLII/TNA result pages) and JSON (CourtListener), and
    tolerates the spacing variation reporters show in the wild (``F. Supp. 3d``).
    """
    import re

    pat = re.compile(r"\s+".join(re.escape(tok) for tok in citation.split()), re.IGNORECASE)
    return bool(pat.search(body or ""))


def http_get(url: str, timeout: int = 20) -> "tuple[int, str]":
    """Default network fetch. Callers wrap exceptions into a fail-closed Resolution."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (trusted gov/LII hosts)
        status = getattr(resp, "status", None) or resp.getcode()
        body = resp.read().decode("utf-8", "replace")
    return status, body
