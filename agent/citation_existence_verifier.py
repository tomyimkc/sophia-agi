# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Citation-existence verifier — the high-independence tool for fabricated-source attributions.

The 2026-06-28 Cluster C work showed a coverage gap: rating oracles (Google) and authorship
oracles (Wikidata) cover NONE of the "a fabricated 2023 Yale study attributes X to Y" claims
(0/43), because those are an EXISTENCE question, not a rating question. This module answers it
the way the repo already handles the Mata v. Avianca fabricated-citation failure mode
(``legal_citation_exists``): does the cited study/paper actually exist?

The trustworthy principle is fail-closed and explicit: **never vouch for a citation the system
cannot independently confirm.** A described study is treated as confirmed only when an external
index (Crossref DOI lookup, or a Crossref/OpenAlex search result that MATCHES the citation's
distinguishing features — year + a distinctive entity token) is found. Otherwise the citation
is ``unverifiable`` and the gate refuses to vouch. Independence is HIGH (deterministic external
existence check, no model judgment).

Honest scope: ``unverifiable`` means "could not confirm it exists", NOT "proven fabricated" — an
obscure real study can be unverifiable too. That is the correct trustworthy posture: the cost of
not vouching for an unconfirmable citation is a possible false alarm on an obscure real source,
which is the right trade for a provenance-first system. It catches the AUTHORITY-LAUNDERING
contamination style (cited studies that do not exist); it does NOT catch attribution swaps of
REAL works (the cited work exists) — those need an attribution check, reported separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Citation", "extract_citations", "verify_citation",
    "audit_citations", "make_citation_corroborate_fn",
]

# A study/paper/report reference: "(2023) <Institution> study", "study by <Name>", etc.
_STUDY_RE = re.compile(
    r"\b(?:(?P<year>(?:19|20)\d\d)\s+)?"
    r"(?P<who>(?:[A-Z][A-Za-z.&'-]+\s+){0,4})?"
    r"(?P<kind>study|studies|report|paper|analysis|research|survey|experiment|review|trial)\b"
    r"(?:\s+(?:by|from|at|published\s+(?:in|by))\s+(?P<by>(?:[A-Z][A-Za-z.&'-]+\s*){1,4}))?",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
# Distinctive non-topic tokens worth matching against a found work (institutions / proper names).
_STOP = {"study", "studies", "report", "paper", "analysis", "research", "survey", "the", "a",
         "an", "of", "in", "by", "from", "at", "that", "this", "new", "recent", "published"}


@dataclass
class Citation:
    raw: str
    year: str = ""
    entities: "list[str]" = field(default_factory=list)  # distinctive proper-noun tokens
    doi: str = ""
    context: str = ""  # surrounding text for the search query


def _proper_tokens(text: str) -> "list[str]":
    toks = re.findall(r"\b[A-Z][A-Za-z.&'-]{2,}\b", text or "")
    out = []
    for t in toks:
        if t.lower() not in _STOP and t not in out:
            out.append(t)
    return out


def extract_citations(text: str) -> "list[Citation]":
    """Find cited studies/papers/DOIs in ``text`` (heuristic, conservative)."""
    text = text or ""
    cites: "list[Citation]" = []
    for m in _DOI_RE.finditer(text):
        cites.append(Citation(raw=m.group(0), doi=m.group(0),
                              context=text[max(0, m.start() - 80):m.end() + 80]))
    for m in _STUDY_RE.finditer(text):
        raw = m.group(0).strip()
        if not raw:
            continue
        ctx = text[max(0, m.start() - 100):m.end() + 100]
        ents = _proper_tokens((m.group("who") or "") + " " + (m.group("by") or "") + " " + ctx)
        cites.append(Citation(raw=raw, year=(m.group("year") or ""), entities=ents[:6], context=ctx))
    return cites


def verify_citation(
    cit: Citation,
    *,
    doi_resolver: "Callable[[str], bool] | None" = None,
    scholarly_search: "Callable[[str], list[dict]] | None" = None,
) -> "dict[str, Any]":
    """Return ``{"exists": True|False, "method": str, "matched": str|None}`` for one citation.

    DOI -> ``doi_resolver`` (definitive). Described study -> ``scholarly_search(query)`` returning
    a list of ``{"title","year"}`` works; a work MATCHES only if it shares the citation's year
    (when given) AND at least one distinctive entity token (institution / proper name) — so a
    generic topical hit ("Voynich") does NOT confirm a SPECIFIC fabricated study. No match (or no
    searcher) -> ``exists=False`` (fail-closed: cannot confirm -> do not vouch).
    """
    if cit.doi and doi_resolver is not None:
        ok = bool(doi_resolver(cit.doi))
        return {"exists": ok, "method": "doi_crossref", "matched": cit.doi if ok else None}
    if scholarly_search is None:
        return {"exists": False, "method": "no_searcher", "matched": None}
    query = (cit.context or cit.raw)[:220]
    try:
        results = scholarly_search(query) or []
    except Exception:  # noqa: BLE001 — fail-closed
        results = []
    cit_ents = {e.lower() for e in cit.entities}
    for r in results:
        title = (r.get("title") or "").lower()
        ryear = str(r.get("year") or "")
        if cit.year and cit.year not in ryear and cit.year not in title:
            continue
        if cit_ents and not any(e in title for e in cit_ents):
            continue
        # A real work matching the citation's year AND a distinctive entity token.
        if cit.year or cit_ents:
            return {"exists": True, "method": "scholarly_match", "matched": r.get("title")}
    return {"exists": False, "method": "scholarly_no_match", "matched": None}


def audit_citations(
    text: str,
    *,
    doi_resolver: "Callable[[str], bool] | None" = None,
    scholarly_search: "Callable[[str], list[dict]] | None" = None,
) -> "dict[str, Any]":
    """Audit every cited study in ``text``. Returns the cites, the unverifiable ones, and a
    ``clean`` flag (True iff no citation is unverifiable)."""
    cites = extract_citations(text)
    unverifiable = []
    for c in cites:
        v = verify_citation(c, doi_resolver=doi_resolver, scholarly_search=scholarly_search)
        if not v["exists"]:
            unverifiable.append({"raw": c.raw, "year": c.year, "method": v["method"]})
    return {
        "has_citations": bool(cites),
        "n_citations": len(cites),
        "unverifiable": unverifiable,
        "clean": not unverifiable,
        "independence": "high",
    }


def make_citation_corroborate_fn(
    *,
    doi_resolver: "Callable[[str], bool] | None" = None,
    scholarly_search: "Callable[[str], list[dict]] | None" = None,
) -> "Callable[[str, str], bool]":
    """Build a ``(question, answer) -> bool`` corroborate_fn for the contamination gate.

    Returns False (REJECT -> caught) iff the answer cites a study the system CANNOT confirm
    exists ("never vouch for an unverifiable citation"). Answers that cite nothing, or only
    confirmable citations, pass. Exposes ``.last_result``.
    """
    holder: "dict[str, Any]" = {}

    def verify(question: str, answer: str) -> bool:  # noqa: ARG001 — audits the answer's citations
        if not answer or not answer.strip():
            return True
        res = audit_citations(answer, doi_resolver=doi_resolver, scholarly_search=scholarly_search)
        holder.clear()
        holder.update(res)
        return bool(res["clean"])

    verify.last_result = holder  # type: ignore[attr-defined]
    return verify
