"""Legal-authority sources: verify citations against primary sources, fail-closed.

    from agent.legal_sources import make_resolver
    from agent.verifiers import legal_citation_exists

    resolver = make_resolver(mode="live")          # or SOPHIA_LEGAL_SOURCE=live
    verifier = legal_citation_exists(resolver=resolver)

See docs/08-Domains/Legal-Industry-Fit.md (connector adoption).
"""

from __future__ import annotations

from agent.legal_sources.base import LegalSource, Resolution
from agent.legal_sources.cache import ResolutionCache
from agent.legal_sources.courtlistener import CourtListenerSource
from agent.legal_sources.elegislation import ELegislationSource
from agent.legal_sources.hklii import HKLIISource
from agent.legal_sources.registry import LegalResolver, make_resolver, resolver_mode
from agent.legal_sources.tna import TNASource

__all__ = [
    "LegalSource", "Resolution", "ResolutionCache", "ELegislationSource",
    "HKLIISource", "TNASource", "CourtListenerSource",
    "LegalResolver", "make_resolver", "resolver_mode",
]
