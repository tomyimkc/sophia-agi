# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Source ranking for grounded gateway verification.

Grounding is not just "has a source"; source quality matters.  This small,
deterministic scorer ranks source identifiers by provenance strength before the
contract sees them.  It is intentionally conservative and offline: no network
lookup, only URI/domain/path heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class RankedSource:
    id: str
    rank: float
    tier: str
    reason: str

    def to_dict(self) -> dict:
        return {"id": self.id, "rank": self.rank, "tier": self.tier, "reason": self.reason}


# High-trust domains for claims that need live/web evidence.  Keep this list
# short and explainable; callers can still pass local OKF/data/docs sources.
_AUTHORITY_DOMAIN_HINTS = (
    ".gov", ".edu", ".ac.", "courtlistener.com", "legislation.gov.uk",
    "legislation.gov.hk", "hklii.hk", "arxiv.org", "doi.org", "pubmed.ncbi.nlm.nih.gov",
    "plato.stanford.edu", "iep.utm.edu", "wikisource.org",
)


def _source_id(src) -> str:
    if isinstance(src, str):
        return src.strip()
    if isinstance(src, dict):
        return str(src.get("id") or src.get("url") or "").strip()
    return ""


def rank_source(src) -> RankedSource:
    """Return a deterministic trust rank in [0, 1] for a source id/url.

    The rank is an evidence-quality prior, not a truth guarantee. Verification
    still happens downstream; this only prevents low-quality / model-only
    strings from being treated as adequate grounding.
    """
    sid = _source_id(src)
    low = sid.lower()
    if not sid:
        return RankedSource(sid, 0.0, "none", "empty source")
    if low.startswith(("okf://", "belief://")):
        return RankedSource(sid, 0.95, "canonical-local", "OKF/belief graph")
    if low.startswith(("data/", "docs/", "wiki/", "benchmark/reference/")):
        return RankedSource(sid, 0.90, "curated-local", "version-controlled corpus path")
    if low.startswith(("file://", "repo://")):
        return RankedSource(sid, 0.82, "local-file", "local repository/file evidence")
    if low.startswith(("wikidata://", "wikisource://", "wiki://")):
        return RankedSource(sid, 0.78, "structured-reference", "structured reference source")
    if low.startswith(("skillforge://", "synthesized:", "gateway://", "tool://")):
        return RankedSource(sid, 0.70, "verified-system", "system-generated but provenance-stamped")
    if low.startswith(("http://", "https://")):
        host = (urlparse(sid).hostname or "").lower()
        if any(h in host for h in _AUTHORITY_DOMAIN_HINTS) or any(host.endswith(h) for h in (".gov", ".edu")):
            return RankedSource(sid, 0.86, "authoritative-web", f"authority domain: {host}")
        return RankedSource(sid, 0.60, "web", f"web source: {host or 'unknown host'}")
    if low.startswith(("model:", "llm:", "chat:")):
        return RankedSource(sid, 0.20, "model-only", "model output is not evidence")
    # Backward-compatible floor for existing offline tests and simple templates:
    # a non-empty explicit source is weak but usable for low-risk claims.
    return RankedSource(sid, 0.55, "generic", "explicit source id, unknown tier")


def rank_sources(sources: list, *, min_rank: float = 0.5) -> dict:
    """Rank and filter sources for grounded verification.

    Returns {accepted, rejected, topRank, minRank}. ``accepted`` preserves the
    original source shape so the governance contract can normalize it.
    """
    ranked = [rank_source(s) for s in (sources or [])]
    accepted = [s for s, r in zip(sources or [], ranked) if r.rank >= min_rank]
    rejected = [r.to_dict() for r in ranked if r.rank < min_rank]
    return {
        "accepted": accepted,
        "rejected": rejected,
        "ranked": [r.to_dict() for r in ranked],
        "topRank": max((r.rank for r in ranked), default=0.0),
        "minRank": min_rank,
    }
