# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Open-world truth-reference retrieval for the independent-verifier defense.

The source-contamination defense (``agent.source_verifier.make_independent_verifier``)
is only as strong as the INDEPENDENCE of its truth-references from the contaminated
grounding source. The structured pack curates those refs by construction, so
independence is *assumed*, not *measured* (see THEORY-ISSUES-FROM-LIVE-RUNS-2026-06-28
issue 5). This module closes that gap for a real run: it fetches short factual summaries
for an entity from an EXTERNAL source (Wikipedia's REST summary API) that is genuinely
independent of the adversarially-contaminated source under test.

Fail-closed: if retrieval returns nothing usable, ``fetch_truth_refs`` returns ``[]``.
The caller then has NO independent reference and the verifier abstains — an empty result
must never be mistaken for "verified". Network access is via an INJECTED ``fetch_fn`` so
tests are deterministic (mock it) while production passes a real urllib-based fetcher.

Honest scope: Wikipedia is itself fallible and can be vandalized; this measures
independence-from-the-source-under-test, not ground truth. It is a stronger, open-world
signal than curated refs — not an oracle.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Callable

__all__ = ["fetch_truth_refs", "default_fetch_fn", "WIKI_SUMMARY_API"]

WIKI_SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# A polite, identifying UA — the Wikimedia REST API asks clients to set one.
_USER_AGENT = "sophia-agi-source-contamination-bench/1.0 (independent truth-ref retrieval)"


def default_fetch_fn(url: str, *, timeout: float = 15.0) -> str | None:
    """Production fetcher: GET ``url`` over HTTPS and return the body text, or None.

    Fail-closed: any network/HTTP error returns None (caller treats as no reference).
    Injected into ``fetch_truth_refs`` only for live runs; tests pass a mock instead.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — https only
            return response.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 — fail-closed: a fetch error is no reference, not a pass
        return None


def _title_to_path(entity: str) -> str:
    """Turn a Wikipedia page title into the REST summary path segment."""
    from urllib.parse import quote

    return quote(entity.strip().replace(" ", "_"), safe="")


def _parse_summary(body: str | None) -> str | None:
    """Extract a short factual summary string from a REST summary JSON body.

    Prefers ``extract`` (the plain-text lead summary); falls back to the HTML-free
    ``description``. Returns None when the body is missing, unparseable, a 404/missing
    page, or has no usable text — all of which are fail-closed (no reference)."""
    if not body:
        return None
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    # REST API marks not-found pages with type "...not_found" / a "missing" flag.
    if data.get("type", "").endswith("not_found") or data.get("missing"):
        return None
    extract = data.get("extract")
    if isinstance(extract, str) and extract.strip():
        return extract.strip()
    description = data.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return None


def fetch_truth_refs(
    entity: str,
    n: int = 2,
    fetch_fn: "Callable[..., str | None] | None" = None,
) -> "list[str]":
    """Retrieve up to ``n`` short factual summaries for ``entity`` from Wikipedia.

    Args:
        entity: a Wikipedia page title (e.g. "Voynich manuscript", "Great Wall of China").
        n: max number of reference texts to return (the REST summary endpoint yields one
            lead summary per title; we split it into up to ``n`` sentence-level refs so the
            ``fact_check_gate`` >=2-independent-domain floor can be satisfied from one page).
        fetch_fn: ``(url) -> body_str | None``. Injected so tests are deterministic; pass
            ``default_fetch_fn`` (or omit, which uses it) for a live HTTPS run.

    Returns:
        A list of up to ``n`` non-empty reference strings, or ``[]`` if retrieval yielded
        nothing usable. FAIL-CLOSED: an empty list signals "no independent reference",
        which the caller MUST treat as a reason to abstain — never as verification.
    """
    if not entity or not entity.strip():
        return []
    fetch = fetch_fn or default_fetch_fn
    url = WIKI_SUMMARY_API + _title_to_path(entity)
    summary = _parse_summary(fetch(url))
    if not summary:
        return []
    refs = _split_into_refs(summary, n)
    return refs[:n] if refs else []


def _split_into_refs(summary: str, n: int) -> "list[str]":
    """Split a lead summary into up to ``n`` distinct sentence-level reference texts.

    One Wikipedia summary is a single source; for the >=2-domain floor we surface its
    first ``n`` sentences as distinct refs (each retains the entity context of the lead).
    If the summary has fewer sentences than ``n``, the whole summary is one ref so a
    short page still yields a (single) usable reference rather than nothing."""
    import re

    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
    if len(parts) >= n >= 1:
        return parts[:n]
    return [summary.strip()]
