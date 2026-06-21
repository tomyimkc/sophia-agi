"""Citation → source routing with a cache-first, fail-closed resolver.

Mode is set by ``SOPHIA_LEGAL_SOURCE``:

- ``off``    — no live/cached resolution; the verifier uses its static register only.
- ``cache``  — (default) cache-first, **no network**: a cache miss is UNVERIFIED.
- ``live``   — cache-first, then hit HKLII / e-Legislation on a miss and cache it.

``make_resolver()`` returns ``(citation) -> Resolution`` for the verifier, or
``None`` in ``off`` mode (meaning "static register only").
"""

from __future__ import annotations

import os
from typing import Callable

from agent.legal_citations import normalize_citation
from agent.legal_sources.base import Fetch, Resolution, http_get, unverified
from agent.legal_sources.cache import ResolutionCache
from agent.legal_sources.elegislation import ELegislationSource
from agent.legal_sources.hklii import HKLIISource

MODES = {"off", "cache", "live"}


def default_sources() -> list:
    return [ELegislationSource(), HKLIISource()]


class LegalResolver:
    def __init__(
        self,
        *,
        mode: str = "cache",
        sources: "list | None" = None,
        cache: "ResolutionCache | None" = None,
        fetch: "Fetch | None" = None,
        timeout: int = 20,
        max_age_days: "int | None" = None,
    ) -> None:
        self.mode = mode if mode in MODES else "cache"
        self.sources = sources if sources is not None else default_sources()
        self.cache = cache if cache is not None else ResolutionCache()
        self.fetch = fetch or http_get
        self.timeout = timeout
        self.max_age_days = max_age_days

    def resolve(self, citation: str) -> Resolution:
        norm = normalize_citation(citation)
        cached = self.cache.get(norm, max_age_days=self.max_age_days)
        if cached is not None:
            return cached
        if self.mode != "live":
            return unverified(norm, "cache", "offline", "no cached verification (cache-only mode)")
        for source in self.sources:
            if source.can_resolve(norm):
                result = source.resolve(norm, fetch=self.fetch, timeout=self.timeout)
                # Cache verified results (and definitive not_found) so we stay polite;
                # transient errors are NOT cached so a retry can succeed.
                if result.status in ("verified", "not_found"):
                    self.cache.put(result)
                    self.cache.save()
                return result
        return unverified(norm, "none", "unsupported", "no source handles this citation shape")


def resolver_mode(explicit: "str | None" = None) -> str:
    return (explicit or os.environ.get("SOPHIA_LEGAL_SOURCE") or "cache").strip().lower()


def make_resolver(mode: "str | None" = None, **kwargs) -> "Callable[[str], Resolution] | None":
    """Build a resolver callable for the verifier, or None for static-only (off)."""
    resolved_mode = resolver_mode(mode)
    if resolved_mode == "off":
        return None
    return LegalResolver(mode=resolved_mode, **kwargs).resolve
