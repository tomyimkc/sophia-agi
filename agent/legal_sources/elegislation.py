# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""e-Legislation source — verifies HK ordinance chapter references (``Cap. 614``).

Hong Kong e-Legislation (elegislation.gov.hk) is, since Q1 2025, the sole official
source of verified consolidated legislation (Legislation Publication Ordinance,
Cap. 614). Ordinances have stable per-chapter URLs, so "does Cap. N exist?" is a
clean existence check.

NOTE: confirm the live URL scheme before relying on this in production — it is
overridable via ``SOPHIA_ELEGISLATION_BASE`` precisely so the architecture does
not hard-depend on an unverified path. The default below is the documented
chapter-page shape; treat it as a best-effort default, not a guarantee.
"""

from __future__ import annotations

import os
import re
import urllib.error

from agent.legal_citations import normalize_citation
from agent.legal_sources.base import Fetch, Resolution, unverified, verified

_CAP = re.compile(r"^Cap\.\s*(\d+[A-Z]?)$")
_DEFAULT_BASE = "https://www.elegislation.gov.hk/hk/cap"


class ELegislationSource:
    name = "elegislation"

    def __init__(self, base: "str | None" = None) -> None:
        self.base = (base or os.environ.get("SOPHIA_ELEGISLATION_BASE") or _DEFAULT_BASE).rstrip("/")

    def can_resolve(self, citation: str) -> bool:
        return bool(_CAP.match(normalize_citation(citation)))

    def url_for(self, cap: str) -> str:
        return f"{self.base}{cap.lower()}"

    def resolve(self, citation: str, *, fetch: Fetch, timeout: int = 20) -> Resolution:
        norm = normalize_citation(citation)
        m = _CAP.match(norm)
        if not m:
            return unverified(norm, self.name, "unsupported", "not an ordinance chapter reference",
                              source_type="legislation")
        cap = m.group(1)
        url = self.url_for(cap)
        try:
            status, body = fetch(url, timeout)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return unverified(norm, self.name, "error", f"fetch failed: {exc}", source_type="legislation")
        if status == 200 and body.strip():
            title = _extract_title(body)
            return verified(norm, self.name, source_type="legislation", url=url, title=title)
        return unverified(norm, self.name, "not_found", f"chapter not found (HTTP {status})",
                          source_type="legislation")


def _extract_title(body: str) -> str:
    m = re.search(r"<title>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", m.group(1)).strip()[:200] if m else ""
