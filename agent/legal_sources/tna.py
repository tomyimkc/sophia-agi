# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""UK source — National Archives "Find Case Law" (caselaw.nationalarchives.gov.uk).

The official publisher of UK court judgments from 2001 (the database the *Ayinde*
court endorsed). Handles UK neutral citations (``[2025] EWHC 1383 (Admin)``,
``[2024] UKSC 1``).

Same posture as the HKLII source: cache-first, polite, defensive parsing,
fail-closed. The base URL / query path is overridable (``SOPHIA_TNA_BASE``) so the
architecture does not hard-depend on an unverified endpoint — confirm the live
search scheme before production use.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse

from agent.legal_citations import UK_COURTS, _NEUTRAL, neutral_court, normalize_citation
from agent.legal_sources.base import Fetch, Resolution, loose_contains, unverified, verified

_DEFAULT_BASE = "https://caselaw.nationalarchives.gov.uk"


class TNASource:
    name = "tna"

    def __init__(self, base: "str | None" = None) -> None:
        self.base = (base or os.environ.get("SOPHIA_TNA_BASE") or _DEFAULT_BASE).rstrip("/")

    def can_resolve(self, citation: str) -> bool:
        return neutral_court(citation) in UK_COURTS

    def url_for(self, citation: str) -> str:
        return f"{self.base}/search?" + urllib.parse.urlencode({"query": normalize_citation(citation)})

    def resolve(self, citation: str, *, fetch: Fetch, timeout: int = 20) -> Resolution:
        norm = normalize_citation(citation)
        m = _NEUTRAL.match(norm)
        if not m:
            return unverified(norm, self.name, "unsupported", "not a neutral citation", source_type="case")
        court, year = m.group(2).upper(), m.group(1)
        url = self.url_for(norm)
        try:
            status, body = fetch(url, timeout)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return unverified(norm, self.name, "error", f"fetch failed: {exc}", source_type="case")
        if status != 200:
            return unverified(norm, self.name, "error", f"search failed (HTTP {status})", source_type="case")
        # Match the bare "[YYYY] COURT NUM" core (the division marker is optional in
        # result listings), so "[2025] EWHC 1383" matches "[2025] EWHC 1383 (Admin)".
        core = f"[{year}] {court} {m.group(3)}"
        if loose_contains(body, core):
            return verified(norm, self.name, source_type="case", url=url, court=court, year=year)
        return unverified(norm, self.name, "not_found", "no matching authority in Find Case Law",
                          source_type="case")
