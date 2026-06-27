# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""ClaimReview retriever — wraps the Google Fact Check Tools API as a
``Retriever`` for :mod:`agent.fact_check_gate`'s Layer-2 external grounding.

Why this exists: the repo's binding constraint is third-party independence — every
existing benchmark is self-authored. The Google Fact Check Tools API aggregates
REAL professional fact-check verdicts (ClaimReview markup from AP, Reuters,
Snopes, PolitiFact, AFP, BBC, Full Fact, ...), so it is a genuine external
grounding source for contemporary claims. Wiring it as a retriever lets the
existing ``fact_check_gate`` consult professional fact-checks at runtime — and the
harvested pack (``tools/build_claimreview_pack.py``) gives a third-party-grounded
eval axis for it.

Domain note (honest): the API covers contemporary claims (vaccines, climate,
politics, science misconceptions) and returns ~0 for historical authorship. So
this retriever is for a NEW capability axis (contemporary-claim verification),
NOT validation of the dolphin provenance delta.

A ``Retriever`` here is the callable shape ``fact_check_gate`` expects:
``retriever(claim: str) -> list[Source]`` where each Source has a ``domain`` and
``text`` the gate's entailment check reads. We map each ClaimReview verdict into a
Source whose text states the professional rating (so a contradiction with the
model's endorsement registers as a contradiction at the gate).
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

BASE = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


@dataclass
class Source:
    """Minimal source shape the fact_check_gate entailment layer reads."""

    domain: str
    text: str
    url: str | None = None
    rating: str | None = None  # the raw textualRating, for audit


def _rating_sentence(rating_raw: str, publisher: str, claim: str) -> str:
    """Render a verdict as a sentence the gate's entailment check can read.

    A professional 'False' verdict becomes an explicit contradiction of the
    claim, so a model answer that ENDORSES the claim registers as contradicting
    the retrieved source (-> rejected at Layer 2)."""
    r = (rating_raw or "").strip().lower()
    if any(w in r for w in ("false", "falso", "falsch", "fake", "wrong", "incorrect",
                            "unsupported", "no evidence", "baseless", "debunk", "myth")):
        return (f"Professional fact-checker {publisher} rated this claim FALSE: "
                f'"{claim}" is not supported by the evidence.')
    if any(w in r for w in ("true", "correct", "accurate")):
        return (f"Professional fact-checker {publisher} rated this claim TRUE: "
                f'"{claim}" is supported by the evidence.')
    if any(w in r for w in ("mixed", "partly", "partially", "half")):
        return (f"Professional fact-checker {publisher} rated this claim MIXED / "
                f"partly accurate: {rating_raw}.")
    return f"Professional fact-checker {publisher} reviewed this claim: {rating_raw}."


def search_claims(query: str, api_key: str, *, max_results: int = 5, timeout: int = 15) -> list[dict]:
    """Call the Google Fact Check Tools API; return raw claim rows."""
    params = {"query": query, "key": api_key, "pageSize": max_results}
    url = BASE + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return (json.load(r) or {}).get("claims", [])


def make_claimreview_retriever(api_key: str | None = None) -> Callable[[str], list[Source]]:
    """Build a ``Retriever`` (claim -> list[Source]) backed by the Fact Check API.

    The key is read from ``api_key`` or ``GOOGLE_FACTCHECK_API_KEY`` /
    ``GFC_API_KEY``. Returns an empty list (no grounding) on any error — fail
    soft, never break the gate (consistent with the repo's best-effort grounding
    policy). Offline / no-key -> empty retriever (gate holds, never fabricates).
    """
    key = api_key or os.environ.get("GOOGLE_FACTCHECK_API_KEY") or os.environ.get("GFC_API_KEY")
    if not key:
        def _empty(_claim: str) -> list[Source]:
            return []
        _empty.__doc__ = "ClaimReview retriever: NO API KEY configured (returns no sources)."
        return _empty

    def _retrieve(claim: str) -> list[Source]:
        try:
            rows = search_claims(claim, key)
        except Exception:
            return []
        sources: list[Source] = []
        for c in rows:
            cr = (c.get("claimReview") or [{}])[0]
            pub = (cr.get("publisher") or {}).get("name", "fact-checker")
            site = (cr.get("publisher") or {}).get("site")
            rating = cr.get("textualRating") or ""
            text = c.get("text") or claim
            sources.append(Source(
                domain=site or pub.lower().replace(" ", ""),
                text=_rating_sentence(rating, pub, text),
                url=cr.get("url"),
                rating=rating,
            ))
        return sources

    _retrieve.__doc__ = "ClaimReview retriever: consults the Google Fact Check Tools API."
    return _retrieve


__all__ = ["Source", "make_claimreview_retriever", "search_claims"]
