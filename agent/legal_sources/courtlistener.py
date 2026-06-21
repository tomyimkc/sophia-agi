"""US source — CourtListener (Free Law Project, courtlistener.com).

Free access to 9M+ US opinions with a documented REST API. Handles US reporter
citations (``925 F.3d 1339``, ``576 U.S. 644``, ``678 F. Supp. 3d 443``) — the
shape of the fabricated authorities in *Mata v. Avianca*.

Resolves via the public search API (GET, JSON) and checks the citation appears in
a returned result; cache-first and fail-closed. Base overridable via
``SOPHIA_COURTLISTENER_BASE``. NOTE: unauthenticated requests are rate-limited;
a token (``COURTLISTENER_API_TOKEN``) is recommended for bulk use — wire it into
the fetch layer before production runs. Confirm the live API shape first.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse

from agent.legal_citations import is_us_reporter, normalize_citation
from agent.legal_sources.base import Fetch, Resolution, loose_contains, unverified, verified

_DEFAULT_BASE = "https://www.courtlistener.com"


class CourtListenerSource:
    name = "courtlistener"

    def __init__(self, base: "str | None" = None) -> None:
        self.base = (base or os.environ.get("SOPHIA_COURTLISTENER_BASE") or _DEFAULT_BASE).rstrip("/")

    def can_resolve(self, citation: str) -> bool:
        return is_us_reporter(citation)

    def url_for(self, citation: str) -> str:
        q = urllib.parse.urlencode({"type": "o", "q": f'"{normalize_citation(citation)}"'})
        return f"{self.base}/api/rest/v4/search/?{q}"

    def resolve(self, citation: str, *, fetch: Fetch, timeout: int = 20) -> Resolution:
        norm = normalize_citation(citation)
        if not is_us_reporter(norm):
            return unverified(norm, self.name, "unsupported", "not a US reporter citation", source_type="case")
        url = self.url_for(norm)
        try:
            status, body = fetch(url, timeout)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return unverified(norm, self.name, "error", f"fetch failed: {exc}", source_type="case")
        if status != 200:
            return unverified(norm, self.name, "error", f"search failed (HTTP {status})", source_type="case")
        # The JSON results echo the reporter citation; a loose (whitespace-flexible)
        # match confirms the authority is real in CourtListener.
        if loose_contains(body, norm):
            return verified(norm, self.name, source_type="case", url=url)
        return unverified(norm, self.name, "not_found", "no matching opinion in CourtListener",
                          source_type="case")
