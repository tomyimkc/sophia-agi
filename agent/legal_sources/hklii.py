"""HKLII source — verifies HK/common-law neutral citations (``[2025] HKCFI 808``).

HKLII (hklii.hk) gives free access to HK primary case law (CFA 1997-, CA/CFI
1946-, District Court, tribunals, Privy Council 1861-1997). It is a public-
interest service in the Free Access to Law Movement, so this source is
**cache-first and polite**: one query per uncached citation, defensive parsing,
fail-closed on anything ambiguous.

NOTE: HKLII has no documented public REST API historically, so this resolves via
search-page retrieval + defensive parsing. The base URL and query path are
overridable (``SOPHIA_HKLII_BASE``) so the architecture does not hard-depend on an
unverified endpoint. Confirm the live search scheme and robots.txt before
production use; the existence signal is conservative (must see all citation parts).
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse

from agent.legal_citations import _NEUTRAL, HK_COURTS, neutral_court, normalize_citation
from agent.legal_sources.base import Fetch, Resolution, unverified, verified

_DEFAULT_BASE = "https://www.hklii.hk"


class HKLIISource:
    name = "hklii"

    def __init__(self, base: "str | None" = None) -> None:
        self.base = (base or os.environ.get("SOPHIA_HKLII_BASE") or _DEFAULT_BASE).rstrip("/")

    def can_resolve(self, citation: str) -> bool:
        return neutral_court(citation) in HK_COURTS

    def url_for(self, citation: str) -> str:
        return f"{self.base}/search?" + urllib.parse.urlencode({"q": normalize_citation(citation)})

    def resolve(self, citation: str, *, fetch: Fetch, timeout: int = 20) -> Resolution:
        norm = normalize_citation(citation)
        m = _NEUTRAL.match(norm)
        if not m:
            return unverified(norm, self.name, "unsupported", "not a neutral citation", source_type="case")
        year, court, num = m.group(1), m.group(2).upper(), m.group(3)
        url = self.url_for(norm)
        try:
            status, body = fetch(url, timeout)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return unverified(norm, self.name, "error", f"fetch failed: {exc}", source_type="case")
        if status != 200:
            return unverified(norm, self.name, "error", f"search failed (HTTP {status})", source_type="case")
        # Conservative existence signal: the full normalized citation must appear in
        # a result body. Substring of the bare "[YYYY] COURT NUM" is required, which
        # avoids matching a stray year or court token elsewhere on the page.
        if _citation_present(body, year, court, num):
            return verified(norm, self.name, source_type="case", url=url,
                            title=_extract_case_name(body, norm), court=court, year=year)
        return unverified(norm, self.name, "not_found", "no matching authority in HKLII results",
                          source_type="case")


def _citation_present(body: str, year: str, court: str, num: str) -> bool:
    pat = re.compile(r"\[\s*" + re.escape(year) + r"\s*\]\s*" + re.escape(court) + r"\s+" + re.escape(num) + r"\b",
                     re.IGNORECASE)
    return bool(pat.search(body or ""))


def _extract_case_name(body: str, norm: str) -> str:
    # Best-effort: the text immediately preceding the citation on the result line.
    m = re.search(r"([A-Z][^\n<>]{3,120}?)\s*" + re.escape(norm), body or "")
    return re.sub(r"\s+", " ", m.group(1)).strip()[:160] if m else ""
