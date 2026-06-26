# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sourced fill — promote knowledge-gap stubs into sourced pages, from trusted sources only.

`agent.gap_ingest` materializes a queried-but-ungrounded topic into a ``none_extant`` draft
stub (a known unknown). This module takes the last, deliberately-guarded step: it fills such a
stub with a real page **extracted from a trusted source** via the existing librarian
(`agent.wiki_librarian`). Two independent boundaries keep the charter's no-fabrication rule:

  1. **Allowlist (input).** A source is eligible only if it is operator-curated (lives under the
     trusted sources directory) or clears the existing trust ranking
     (`agent.source_ranking.rank_source` ≥ ``min_trust`` — authority domains, curated-local,
     canonical). An un-allowlisted source (`model:`, a random web host) is **refused before any
     extraction**, so the model can never launder itself into the corpus as a source.
  2. **Provenance gate (output).** The extracted page passes the same hard gate as every agent
     write (`agent.wiki_store.gate`: schema + `provenance_faithful` + doNotAttributeTo), so a
     hallucinated attribution is rejected even from an allowlisted source.

The extraction itself needs NLP, so it is **LLM-gated** (operator-run with a key) — but the
model call is *injectable* (``extractor``), so the loop, the allowlist, and the gating are fully
deterministic and offline-testable. A filled stub is written to the quarantined ``draft`` tier
and keeps ``needsReview: true`` — sourced, but still awaiting human sign-off before canon.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from agent.source_ranking import rank_source

#: Minimum trust rank for a non-local source id to be allowlisted (authority web / curated).
DEFAULT_MIN_TRUST = 0.8
#: Source-id prefixes treated as operator-curated (trusted by placement, not by ranking).
DEFAULT_TRUSTED_PREFIXES = ("raw/", "okf://", "data/", "docs/", "wiki/", "benchmark/reference/")
#: A stub is fillable iff it is an auto-generated, unsourced knowledge-gap page.
_STUB_PROVENANCE = "knowledge_gap"
_STUB_CONFIDENCE = "none_extant"

_WORD_RE = re.compile(r"[a-z0-9一-鿿]+")


@dataclass
class TrustedSource:
    id: str        # allowlist/display id, e.g. "raw/penicillin_history.txt"
    name: str      # basename stem passed to the librarian (-> sources: ["raw/<name>"])
    text: str


def is_allowlisted(
    source_id: str, *, min_trust: float = DEFAULT_MIN_TRUST,
    allowlist: "tuple[str, ...]" = (), trusted_prefixes: "tuple[str, ...]" = DEFAULT_TRUSTED_PREFIXES,
) -> bool:
    """True iff ``source_id`` is operator-curated, explicitly allowed, or clears the trust rank."""
    sid = (source_id or "").strip()
    if not sid:
        return False
    if sid in allowlist:
        return True
    low = sid.lower()
    if any(low.startswith(p) for p in trusted_prefixes):
        return True
    return rank_source(sid).rank >= min_trust


def load_trusted_sources(
    sources_dir, *, namespace: str = "raw", patterns: "tuple[str, ...]" = ("*.txt", "*.md")
) -> "list[TrustedSource]":
    """Load operator-curated source files from ``sources_dir`` (non-recursive over patterns)."""
    root = Path(sources_dir)
    if not root.exists():
        return []
    out: list[TrustedSource] = []
    seen: set[str] = set()
    for pat in patterns:
        for path in sorted(root.glob(pat)):
            if path.name in seen or not path.is_file():
                continue
            seen.add(path.name)
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                out.append(TrustedSource(id=f"{namespace}/{path.name}", name=path.stem, text=text))
    return out


def _tokens(text: str) -> "set[str]":
    return {t for t in _WORD_RE.findall((text or "").lower()) if len(t) > 2}


def match_source(stub_id: str, queries: "list[str]", sources: "list[TrustedSource]",
                 *, min_overlap: int = 1) -> "TrustedSource | None":
    """Best trusted source for a stub by token overlap of its id+queries against source id+text.

    Deterministic: highest overlap wins, ties broken by source id order. Returns None below
    ``min_overlap`` so an unrelated source is never force-fit onto a stub.
    """
    needle = _tokens(stub_id.replace("_", " ")) | _tokens(" ".join(queries or []))
    best, best_score = None, 0
    for src in sources:
        hay = _tokens(src.id.replace("/", " ").replace("_", " ")) | _tokens(src.text[:2000])
        score = len(needle & hay)
        if score > best_score or (score == best_score and best is not None and src.id < best.id):
            if score > best_score:
                best, best_score = src, score
    return best if best_score >= min_overlap else None


def make_llm_extractor(client=None):
    """Default extractor: the librarian's LLM page extraction (operator-run; needs a key).

    Returns ``extract(source_text, source_id) -> proposal_dict | None``, mirroring
    ``wiki_librarian.ingest_text`` but returning the proposal so the caller can pin the id.
    """
    from agent import untrusted
    from agent.model import default_client
    from agent.wiki_librarian import LIBRARIAN_SYSTEM, _extract_json

    client = client or default_client()

    def extract(source_text: str, source_id: str):
        fenced = untrusted.wrap_untrusted(source_text, f"raw:{source_id}")
        user = (
            f"Source id: {source_id}\n\n{fenced}\n\nOutput ONLY the JSON page object. Apply source "
            "discipline: assert no attribution the source does not support; set doNotAttributeTo "
            "for known traps."
        )
        result = client.generate(LIBRARIAN_SYSTEM, user)
        if not getattr(result, "ok", False):
            return None
        return _extract_json(result.text)

    return extract


def fill_stub(
    stub_id: str, *, queries: "list[str]", sources: "list[TrustedSource]", extractor,
    write: bool = False, tier: str = "draft", min_trust: float = DEFAULT_MIN_TRUST,
    allowlist: "tuple[str, ...]" = (),
) -> dict:
    """Try to fill one stub from a matching trusted source. Fail-closed on every guard.

    Refuses (never raises) when: no matching source, the source is not allowlisted, or the
    extractor yields nothing. On a successful extraction the page is **pinned to ``stub_id``**
    (so the stub is promoted, not duplicated), stamped ``provenance: librarian_fill`` +
    ``needsReview: true``, and — when ``write`` — gate-written to the draft tier.
    """
    src = match_source(stub_id, queries, sources)
    if src is None:
        return {"id": stub_id, "ok": False, "reason": "no matching trusted source"}
    if not is_allowlisted(src.id, min_trust=min_trust, allowlist=allowlist):
        return {"id": stub_id, "ok": False, "reason": "source not allowlisted", "source": src.id,
                "rank": rank_source(src.id).rank}

    proposal = extractor(src.text, src.id)
    if not proposal or not isinstance(proposal, dict):
        return {"id": stub_id, "ok": False, "reason": "extractor produced no page", "source": src.id}

    # Pin the id so we PROMOTE the stub (upsert merges onto it), not duplicate it.
    proposal["id"] = stub_id

    from agent import wiki_librarian, wiki_store

    # Reuse the librarian's pure proposal→(meta,body) transform, then stamp fill provenance so
    # the promotion is recorded and the page still demands human sign-off before canon.
    meta, body = wiki_librarian.build_page(proposal, src.name)
    meta["provenance"] = "librarian_fill"
    meta["needsReview"] = True

    if not write:
        # Merge onto any existing stub exactly as upsert would, then gate — a true dry-run verdict.
        existing = wiki_store.read_page(stub_id)
        merged = dict(existing.meta) if existing else {}
        merged.update(meta)
        merged.setdefault("pageType", "concept")
        ok, reasons = wiki_store.gate(merged, body)
        return {"id": stub_id, "ok": ok, "wouldFill": ok, "source": src.id,
                "reasons": reasons, "authorConfidence": merged.get("authorConfidence")}

    result = wiki_store.upsert(stub_id, meta=meta, body=body, tier=tier)
    return {"id": stub_id, "ok": bool(result.get("ok")), "source": src.id,
            "filled": bool(result.get("ok")), "reasons": result.get("reasons"),
            "path": result.get("path"), "authorConfidence": meta.get("authorConfidence")}


def is_fillable_stub(page) -> bool:
    """A page is a fillable knowledge-gap stub (auto-generated, unsourced)."""
    meta = getattr(page, "meta", {}) or {}
    return meta.get("provenance") == _STUB_PROVENANCE and meta.get("authorConfidence") == _STUB_CONFIDENCE


def fill_gaps(
    pages, sources, *, extractor, write: bool = False, tier: str = "draft",
    min_trust: float = DEFAULT_MIN_TRUST, allowlist: "tuple[str, ...]" = (),
) -> dict:
    """Attempt to fill every fillable stub among ``pages`` from ``sources``. Returns a report."""
    stubs = [p for p in pages if is_fillable_stub(p)]
    results = []
    for stub in stubs:
        queries = _stub_queries(stub)
        results.append(fill_stub(stub.id, queries=queries, sources=sources, extractor=extractor,
                                 write=write, tier=tier, min_trust=min_trust, allowlist=allowlist))
    filled = [r for r in results if r.get("ok")]
    return {
        "schema": "sophia.source_fill.report.v1",
        "candidateOnly": True,
        "wrote": bool(write),
        "stubs": len(stubs),
        "trustedSources": len(sources),
        "filledOrWould": len(filled),
        "results": results,
    }


def _stub_queries(page) -> "list[str]":
    """Recover the triggering queries recorded in a gap stub's body (best-effort)."""
    body = getattr(page, "body", "") or ""
    out: list[str] = []
    capture = False
    for line in body.splitlines():
        if line.startswith("## Queries"):
            capture = True
            continue
        if capture and line.startswith("- "):
            out.append(line[2:].strip())
    return out


__all__ = [
    "DEFAULT_MIN_TRUST", "DEFAULT_TRUSTED_PREFIXES", "TrustedSource", "fill_gaps", "fill_stub",
    "is_allowlisted", "is_fillable_stub", "load_trusted_sources", "make_llm_extractor",
    "match_source",
]
