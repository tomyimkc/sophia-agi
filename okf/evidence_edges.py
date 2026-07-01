# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic typed-evidence-edge MINER over OKF pages.

Proposes typed edges among wiki pages using ONLY reproducible signals (shared
domain, shared tradition, shared subfield, canonicalTitle token overlap, shared
sources, and existing data/attributions.json relationships). It is a *proposal*
engine: it never mutates any page. The relation vocabulary is drawn from the
repo's existing edge language — supports / refines / relatedTo / sameTradition —
and NEVER emits 'merge'/'sameAs'. Identity/merge across vocabularies is the
owl:sameAs bulldozer this codebase exists to prevent.

HARD CONSTRAINTS (enforced structurally, not by convention):
  1. Never link across a declared ``doNotMergeWith`` (in either direction).
  2. Never emit a same-lineage/merge-flavoured edge (``supports``/``refines``/
     ``sameTradition``) touching a PROTECTED domain (religion, history). Those
     pairs may only ever receive a weaker ``relatedTo`` edge.

The miner is fully deterministic: given the same pages it returns the same edge
list in the same order (pairs are canonicalised and the output is sorted). No
randomness, no I/O beyond an optionally-supplied attributions dict.
"""

from __future__ import annotations

import re
from pathlib import Path


def _import_as_list():
    """Import okf.schema.as_list, tolerating a broken sibling in okf/__init__.

    Importing ``okf.schema`` normally executes ``okf/__init__.py``, which (in some
    environments) transitively imports agent modules that use Python-3.11+ regex
    syntax and fail to compile under older interpreters. To stay robust and
    dependency-honest we try the normal import first, then fall back to loading
    ``okf/schema.py`` directly by file path (schema.py itself is pure stdlib).
    """
    try:
        from okf.schema import as_list as _al
        return _al
    except Exception:  # pragma: no cover - only on a broken sibling / old runtime
        import importlib.util
        schema_path = Path(__file__).resolve().parent / "schema.py"
        spec = importlib.util.spec_from_file_location("okf_schema_direct", schema_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.as_list


as_list = _import_as_list()

# Domains this repo treats as protected — same-lineage/merge reasoning across
# them is forbidden by the project charter; they may only get 'relatedTo'.
PROTECTED_DOMAINS = frozenset({"religion", "history"})

# Relation kinds, in DESCENDING strength. 'sameAs'/'merge' are intentionally
# absent — proposing an identity edge is out of scope for a miner.
RELATION_KINDS = ("supports", "refines", "sameTradition", "relatedTo")

# Merge-flavoured kinds: those that assert lineage/identity affinity and so are
# forbidden when either endpoint is in a PROTECTED domain.
MERGE_FLAVOURED = frozenset({"supports", "refines", "sameTradition"})

# Tokens too generic to count as canonicalTitle overlap evidence.
_STOP_TOKENS = frozenset({
    "the", "a", "an", "of", "in", "on", "and", "or", "to", "by", "for",
    "concept", "structure", "theory", "effect", "bias", "religion", "ritual",
})

# Weight each independent signal contributes to an edge score. Deterministic.
_SIGNAL_WEIGHTS = {
    "shared_domain": 0.15,
    "shared_tradition": 0.35,
    "shared_subfield": 0.25,
    "title_token_overlap": 0.30,
    "shared_source": 0.40,
    "attribution_relationship": 0.30,
    "shared_attributed_author": 0.20,
}


def _norm(value) -> str:
    return str(value).strip().lower() if value is not None else ""


def _title_tokens(meta: dict) -> "set[str]":
    """Content tokens from the English canonical title (stopwords removed)."""
    title = _norm(meta.get("canonicalTitleEn"))
    if not title:
        return set()
    raw = re.split(r"[^a-z0-9]+", title)
    return {t for t in raw if t and t not in _STOP_TOKENS and len(t) > 2}


def _source_set(meta: dict) -> "set[str]":
    return {_norm(s) for s in as_list(meta.get("sources")) if _norm(s)}


def _dnm_set(meta: dict) -> "set[str]":
    """The page's doNotMergeWith targets (normalised)."""
    return {_norm(x) for x in as_list(meta.get("doNotMergeWith")) if _norm(x)}


def _page_view(page) -> dict:
    """A minimal, hashable view of the fields the miner reads."""
    meta = page.meta if isinstance(getattr(page, "meta", None), dict) else {}
    return {
        "id": _norm(page.id),
        "pageType": _norm(meta.get("pageType")),
        "domain": _norm(meta.get("domain")),
        "tradition": _norm(meta.get("tradition")),
        "subfield": _norm(meta.get("subfield")),
        "attributedAuthor": _norm(meta.get("attributedAuthor")),
        "title_tokens": _title_tokens(meta),
        "sources": _source_set(meta),
        "dnm": _dnm_set(meta),
    }


def _blocked_by_dnm(a: dict, b: dict) -> bool:
    """True if either endpoint declares the other in doNotMergeWith.

    We match a doNotMergeWith entry against the other page's id, its tradition,
    and its domain — any of which is a reason the corpus forbids merging them.
    """
    a_targets = {a["id"], a["tradition"], a["domain"]} - {""}
    b_targets = {b["id"], b["tradition"], b["domain"]} - {""}
    if a["dnm"] & b_targets:
        return True
    if b["dnm"] & a_targets:
        return True
    return False


def _touches_protected(a: dict, b: dict) -> bool:
    return a["domain"] in PROTECTED_DOMAINS or b["domain"] in PROTECTED_DOMAINS


def _attribution_signals(a: dict, b: dict, attributions: dict) -> "list[str]":
    """Signals derived from data/attributions.json relationships.

    Reproducible relationships we mine:
      - both textIds appear and share a tradition/domain already recorded there;
      - one page's id is named in the other's attributions doNotAttributeTo
        (this is a NEGATIVE — it never yields a merge edge, only weak related).
    """
    signals: list[str] = []
    ra = attributions.get(a["id"]) if attributions else None
    rb = attributions.get(b["id"]) if attributions else None
    if isinstance(ra, dict) and isinstance(rb, dict):
        ta, tb = _norm(ra.get("tradition")), _norm(rb.get("tradition"))
        if ta and ta == tb:
            signals.append("attribution_relationship")
        aa, ab = _norm(ra.get("attributedAuthor")), _norm(rb.get("attributedAuthor"))
        if aa and aa == ab:
            signals.append("shared_attributed_author")
    return signals


def _raw_signals(a: dict, b: dict, attributions: dict) -> "list[str]":
    """Every reproducible signal present between two page views (ordered)."""
    signals: list[str] = []
    if a["domain"] and a["domain"] == b["domain"]:
        signals.append("shared_domain")
    if a["tradition"] and a["tradition"] == b["tradition"]:
        signals.append("shared_tradition")
    if a["subfield"] and a["subfield"] == b["subfield"]:
        signals.append("shared_subfield")
    if a["title_tokens"] & b["title_tokens"]:
        signals.append("title_token_overlap")
    if a["sources"] & b["sources"]:
        signals.append("shared_source")
    if a["attributedAuthor"] and a["attributedAuthor"] == b["attributedAuthor"]:
        if "shared_attributed_author" not in signals:
            signals.append("shared_attributed_author")
    for sig in _attribution_signals(a, b, attributions):
        if sig not in signals:
            signals.append(sig)
    return signals


def score_edge(signals: "list[str]") -> float:
    """Deterministic score in [0, 1] from a signal list (saturating sum)."""
    total = sum(_SIGNAL_WEIGHTS.get(s, 0.0) for s in signals)
    return round(min(1.0, total), 4)


def _choose_kind(a: dict, b: dict, signals: "list[str]") -> str:
    """Pick the strongest admissible relation kind for a signal set.

    A same-tradition signal proposes 'sameTradition'. A shared-source or
    attribution relationship (co-derivation) proposes 'supports'. A strong title
    overlap within one domain proposes 'refines'. Everything else is 'relatedTo'.
    PROTECTED-domain pairs are DOWNGRADED to 'relatedTo' by the caller — this
    function only proposes the pre-guard kind.
    """
    if "shared_tradition" in signals:
        return "sameTradition"
    if "shared_source" in signals or "attribution_relationship" in signals:
        return "supports"
    if "title_token_overlap" in signals and "shared_domain" in signals:
        return "refines"
    return "relatedTo"


def _make_edge(a: dict, b: dict, signals: "list[str]") -> "dict | None":
    """Build one canonicalised proposed edge, applying HARD CONSTRAINTS.

    Returns None if the pair is blocked (doNotMergeWith) or if there is no
    signal at all. PROTECTED-domain pairs are demoted to 'relatedTo'.
    """
    if not signals:
        return None
    if _blocked_by_dnm(a, b):
        return None
    kind = _choose_kind(a, b, signals)
    if kind in MERGE_FLAVOURED and _touches_protected(a, b):
        kind = "relatedTo"  # HARD CONSTRAINT: no same-lineage edge on protected domain
    # Canonicalise endpoint order so the same pair always yields one edge.
    src, dst = sorted([a["id"], b["id"]])
    return {
        "src": src,
        "dst": dst,
        "kind": kind,
        "evidence": list(signals),
        "score": score_edge(signals),
    }


def mine_edges(pages, *, attributions: "dict | None" = None,
               min_score: float = 0.0) -> "list[dict]":
    """Mine typed evidence edges over OKF pages.

    Deterministic: pairs are enumerated in sorted-id order and the result is
    sorted by (-score, src, dst, kind). No page is mutated. Self-pairs and
    doNotMergeWith / protected-domain violations are structurally excluded.

    Args:
        pages: iterable of okf.Page (or any object with ``.id`` and ``.meta``).
        attributions: optional data/attributions.json mapping (id -> record).
        min_score: drop edges scoring below this floor (default 0 -> keep all
            with >=1 signal).
    """
    attributions = attributions or {}
    views = sorted((_page_view(p) for p in pages), key=lambda v: v["id"])
    edges: list[dict] = []
    seen: set = set()
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            a, b = views[i], views[j]
            if not a["id"] or not b["id"] or a["id"] == b["id"]:
                continue
            key = tuple(sorted([a["id"], b["id"]]))
            if key in seen:
                continue
            signals = _raw_signals(a, b, attributions)
            edge = _make_edge(a, b, signals)
            if edge is None or edge["score"] < min_score:
                continue
            seen.add(key)
            edges.append(edge)
    edges.sort(key=lambda e: (-e["score"], e["src"], e["dst"], e["kind"]))
    return edges


def load_attributions(path) -> dict:
    """Load data/attributions.json if present (else empty dict). Never raises."""
    import json
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _area_of(view: dict) -> str:
    """Approximate a page's top-level area: its domain, else its pageType."""
    return view["domain"] or view["pageType"] or "?"


def coupling_metrics(pages, edges, *, min_signals_for_precise: int = 2,
                     min_edge_score_for_proxy: float = 0.0) -> dict:
    """Compute the coupling / quality metrics the gate reads.

    Metrics:
      edgeDensity              edges / page
      crossThemeCoupling       fraction of edges spanning two distinct areas
                               (area ~ domain, else pageType)
      precisionProxy           fraction of edges (scoring >= floor) whose evidence
                               has >= min_signals_for_precise independent signals.
                               This is the anti-Goodhart quality floor.
      perKind                  count of edges per relation kind.
    All fractions are over the edges considered; deterministic.
    """
    views = {v["id"]: v for v in (_page_view(p) for p in pages)}
    n_pages = len(views)
    n_edges = len(edges)

    cross = 0
    for e in edges:
        a, b = views.get(e["src"]), views.get(e["dst"])
        if a is None or b is None:
            continue
        if _area_of(a) != _area_of(b):
            cross += 1

    proxy_pool = [e for e in edges if e["score"] >= min_edge_score_for_proxy]
    precise = sum(1 for e in proxy_pool
                  if len(e.get("evidence", [])) >= min_signals_for_precise)

    per_kind: dict = {}
    for e in edges:
        per_kind[e["kind"]] = per_kind.get(e["kind"], 0) + 1

    return {
        "pages": n_pages,
        "edges": n_edges,
        "edgeDensity": round(n_edges / n_pages, 4) if n_pages else 0.0,
        "crossThemeEdges": cross,
        "crossThemeCoupling": round(cross / n_edges, 4) if n_edges else 0.0,
        "preciseEdges": precise,
        "proxyPool": len(proxy_pool),
        "precisionProxy": round(precise / len(proxy_pool), 4) if proxy_pool else 0.0,
        "perKind": {k: per_kind.get(k, 0) for k in RELATION_KINDS},
    }
