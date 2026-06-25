# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Canonical entity + sense layer with versioned ids over the OKF belief graph.

A belief that only ever lived in weights has no name you can hold onto: "the same
fact" before and after a revision is whatever the network happens to encode, and
nothing forces the two to coincide. Sophia's discipline is the opposite — every
claim is a page with an id, and the relations between claims (``supersedes`` /
``supersededBy``, ``derivesFrom``, aliases) are written down. This module turns
those written-down relations into a *stable address* for a fact or entity, one
that survives the things that ordinarily make identity slippery:

  * contradiction and supersession — "Pluto is a planet" and "Pluto is a dwarf
    planet" are two *versions* of one underlying claim, not two unrelated facts;
  * retraction and restore — forgetting a source un-grounds a downstream claim,
    but the claim's canonical identity must not move, so that restoring the
    source brings back *the same* belief, bit for bit;
  * time — the temporal-validity layer scopes a fact to a window; identity is the
    orthogonal axis (which fact, regardless of when it is in force).

Three layers, all pure and deterministic (no wall-clock, no randomness):

  1. **Resolution** — ``canonical_id`` / ``resolve_all`` / ``is_ambiguous`` map any
     reference (page id, alias, ``[[wikilink]]``, or an author surface form via
     ``agent.entity_aliases.author_surface_forms``) to canonical node ids. A
     surface form that denotes more than one page is never silently collapsed: the
     ambiguity is surfaced, and the single-id answer is a deterministic (sorted)
     pick, not a guess that hides the collision.
  2. **Sense** — ``build_sense_index`` maps each surface form to the senses it can
     carry, each tagged with a stable discriminator (pageType, domain/tradition),
     so "Plato" the figure and "Plato" some other sense can be told apart by
     context rather than by luck.
  3. **Versioned identity** — ``lineage`` / ``current_version`` / ``stable_identity``
     / ``version_tag`` follow the supersession chain to give every version of a
     claim one invariant canonical key (shared across the whole chain, unchanged
     under retract/restore) plus a content-addressable per-version pin.

``identity_round_trip_report`` proves the invariant end-to-end: it captures every
fact's identity, drives a ``forget`` + ``restore`` cycle through
``agent.unlearning.Unlearner``, and asserts nothing drifted.

    from agent.symbol_identity import stable_identity, lineage
    g = build_graph(pages)
    lineage(g, "pluto_planet")            # ['pluto_planet', 'pluto_dwarf']
    stable_identity(g, "pluto_planet") == stable_identity(g, "pluto_dwarf")  # True
"""

from __future__ import annotations

import hashlib

from okf import build_graph
from okf.graph import Graph, resolve
from okf.schema import DOMAINS, as_list
from okf.wikilinks import normalize_target

# Stable scheme tags baked into the emitted identifiers, so a downstream consumer
# can tell a canonical key / version pin apart by prefix and so a change of scheme
# is a visible, testable change rather than a silent drift.
_STABLE_PREFIX = "okf:claim:"
_VERSION_SCHEME = "v1"


# ---------------------------------------------------------------------------
# Resolution layer
# ---------------------------------------------------------------------------

def _surface_form_candidates(graph: Graph, ref: str) -> "list[str]":
    """Node ids whose author surface forms include the (slugified) ``ref``.

    A *figure* page's identity-bearing display name is its ``attributedAuthor``
    (falling back to a ``name`` meta field, then the id). We expand that name with
    ``agent.entity_aliases.author_surface_forms`` (full name, guarded surname,
    name orderings, known transliterations), slugify each form, and index it to the
    page id. A bare surname like "plato" then resolves to the figure page, and a
    surname shared by two figures resolves to *both* (an exposed ambiguity).
    """
    from agent.entity_aliases import author_surface_forms  # noqa: PLC0415

    want = normalize_target(ref)
    if not want:
        return []
    hits: list[str] = []
    for nid, node in graph.nodes.items():
        if node.get("pageType") != "figure":
            continue
        meta = node["meta"]
        name = meta.get("attributedAuthor") or meta.get("name") or nid
        forms = {normalize_target(f) for f in author_surface_forms(str(name))}
        forms.add(normalize_target(str(name)))
        if want in forms and nid not in hits:
            hits.append(nid)
    return hits


def resolve_all(graph: Graph, ref: str) -> "list[str]":
    """ALL canonical node ids a reference could denote, sorted and deduplicated.

    Tries, in order, exact id / alias resolution (``okf.graph.resolve``) and then
    author surface-form expansion over figure pages. Every distinct hit is kept,
    so a surface form that collides across pages returns more than one id rather
    than committing to one. The result is sorted for determinism.
    """
    if ref is None:
        return []
    ids: set[str] = set()
    direct = resolve(graph, ref)
    if direct is not None:
        ids.add(direct)
    for nid in _surface_form_candidates(graph, ref):
        ids.add(nid)
    return sorted(ids)


def canonical_id(graph: Graph, ref: str) -> "str | None":
    """Resolve a reference to ONE canonical node id, deterministically.

    The single id is the first of ``resolve_all`` under a stable sort, so the
    choice never depends on dict/iteration order or on time. When the reference is
    ambiguous (``is_ambiguous``), this still returns a value — but the collision is
    *not* hidden: a caller that cares must consult ``resolve_all`` /
    ``is_ambiguous`` rather than trusting the lone pick.
    """
    candidates = resolve_all(graph, ref)
    return candidates[0] if candidates else None


def is_ambiguous(graph: Graph, ref: str) -> bool:
    """True iff ``ref`` resolves to more than one canonical node id."""
    return len(resolve_all(graph, ref)) > 1


# ---------------------------------------------------------------------------
# Sense layer
# ---------------------------------------------------------------------------

def _context_of(meta: dict, page_type) -> str:
    """A stable, human-readable discriminator for a sense.

    Combines ``pageType`` with the strongest available domain signal (``domain``,
    else ``tradition``). Deterministic and free of time/randomness, so two runs
    produce byte-identical contexts.
    """
    parts: list[str] = [str(page_type or "unknown")]
    domain = meta.get("domain")
    tradition = meta.get("tradition")
    if domain and str(domain) in DOMAINS:
        parts.append(str(domain))
    elif domain:
        parts.append(str(domain))
    elif tradition:
        parts.append(str(tradition))
    return "/".join(parts)


def _surface_forms_for_page(page) -> "list[str]":
    """Every surface form (slug) under which a page should be discoverable.

    Always its id and declared aliases; for figure pages, also its author surface
    forms (surname etc.). Slugified and order-preserving-deduped.
    """
    from agent.entity_aliases import author_surface_forms  # noqa: PLC0415

    forms: list[str] = [page.id]
    forms.extend(page.aliases)
    if page.page_type == "figure":
        name = page.meta.get("attributedAuthor") or page.meta.get("name") or page.id
        forms.extend(normalize_target(f) for f in author_surface_forms(str(name)))
    seen: set[str] = set()
    out: list[str] = []
    for f in forms:
        slug = normalize_target(str(f))
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


def build_sense_index(pages) -> "dict":
    """Map each surface form to the senses it can carry.

    Returns ``{surface_form: {"senses": [{"id","context"}, ...], "ambiguous": bool}}``
    where ``senses`` is sorted by ``(id, context)`` for determinism and
    ``ambiguous`` flags surface forms that denote more than one page. This is the
    layer that lets a reference be disambiguated *by context* (pageType + domain /
    tradition) instead of by an arbitrary pick — e.g. "plato" the figure vs. any
    other page that also answers to "plato".
    """
    index: dict[str, dict] = {}
    for page in pages:
        context = _context_of(page.meta, page.page_type)
        sense = {"id": page.id, "context": context}
        for form in _surface_forms_for_page(page):
            entry = index.setdefault(form, {"senses": [], "_ids": set()})
            if page.id not in entry["_ids"]:
                entry["_ids"].add(page.id)
                entry["senses"].append(sense)
    out: dict[str, dict] = {}
    for form in sorted(index):
        senses = sorted(index[form]["senses"], key=lambda s: (s["id"], s["context"]))
        out[form] = {"senses": senses, "ambiguous": len(senses) > 1}
    return out


# ---------------------------------------------------------------------------
# Versioned identity / lineage layer
# ---------------------------------------------------------------------------

def _supersedes_targets(graph: Graph, nid: str) -> "list[str]":
    node = graph.nodes.get(nid)
    if node is None:
        return []
    out: list[str] = []
    for raw in as_list(node["meta"].get("supersedes")):
        dep = resolve(graph, raw)
        if dep is not None and dep != nid and dep not in out:
            out.append(dep)
    return out


def _superseded_by_targets(graph: Graph, nid: str) -> "list[str]":
    node = graph.nodes.get(nid)
    if node is None:
        return []
    out: list[str] = []
    for raw in as_list(node["meta"].get("supersededBy")):
        dep = resolve(graph, raw)
        if dep is not None and dep != nid and dep not in out:
            out.append(dep)
    return out


def _walk_oldest(graph: Graph, nid: str) -> str:
    """Follow ``supersedes`` toward the oldest version (lineage root).

    A node *supersedes* the thing it replaced, so the root is reached by walking
    ``supersedes`` until there is none. Ties (more than one superseded target) pick
    the smallest id for determinism; cycles are broken by a visited guard, and the
    smallest id on the cycle is taken as the canonical root.
    """
    seen: set[str] = set()
    current = nid
    while True:
        seen.add(current)
        nxts = [t for t in _supersedes_targets(graph, current) if t not in seen]
        if not nxts:
            # If we stopped because of a cycle, anchor on the smallest id seen.
            cyclic = [t for t in _supersedes_targets(graph, current) if t in seen]
            if cyclic:
                return min(seen)
            return current
        current = sorted(nxts)[0]


def lineage(graph: Graph, ref: str) -> "list[str]":
    """The ordered version chain for a claim, oldest -> newest.

    Resolves ``ref`` to a node, walks to the lineage root via ``supersedes``, then
    walks forward via ``supersededBy`` to the head. A page with no supersession
    links is a 1-element lineage. Branches / ties are ordered by id and cycles are
    guarded, so the chain is deterministic. An unresolvable ``ref`` yields ``[]``.
    """
    nid = canonical_id(graph, ref)
    if nid is None or nid not in graph.nodes:
        return []
    root = _walk_oldest(graph, nid)
    chain: list[str] = [root]
    seen: set[str] = {root}
    current = root
    while True:
        nxts = [t for t in _superseded_by_targets(graph, current)
                if t in graph.nodes and t not in seen]
        if not nxts:
            break
        current = sorted(nxts)[0]
        chain.append(current)
        seen.add(current)
    return chain


def current_version(graph: Graph, ref: str) -> "str | None":
    """The head (newest) of the lineage — the version currently in force."""
    chain = lineage(graph, ref)
    return chain[-1] if chain else None


def stable_identity(graph: Graph, ref: str) -> "str | None":
    """The INVARIANT canonical key for the underlying claim/entity.

    Shared by every version in one supersession chain and unchanged under
    retract/restore (it is derived from graph *structure* — the lineage root — and
    optional explicit overrides, never from grounding state or time).

    Precedence, highest first:
      * an explicit ``meta["canonicalId"]`` on the resolved node;
      * an explicit ``meta["sameAs"]`` (first resolvable target), letting a curator
        pin two pages to one identity even without a supersession edge;
      * otherwise the lineage root id.

    The chosen key is wrapped with a scheme prefix so the identifier is
    self-describing and a scheme change is a visible, testable change.
    """
    nid = canonical_id(graph, ref)
    if nid is None or nid not in graph.nodes:
        return None
    meta = graph.nodes[nid]["meta"]

    explicit = meta.get("canonicalId")
    if explicit:
        return f"{_STABLE_PREFIX}{normalize_target(str(explicit))}"

    same_as = as_list(meta.get("sameAs"))
    if same_as:
        for raw in same_as:
            tgt = resolve(graph, raw)
            if tgt is not None:
                # Anchor on the lineage root of the sameAs target for stability.
                return f"{_STABLE_PREFIX}{_walk_oldest(graph, tgt)}"

    root = _walk_oldest(graph, nid)
    return f"{_STABLE_PREFIX}{root}"


def _short_hash(payload: str) -> str:
    """Deterministic short hash (sha256, hex-sliced). No seeding, no randomness."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def version_tag(graph: Graph, ref: str) -> "str | None":
    """A DETERMINISTIC, content-addressable version pin for a specific fact version.

    Form: ``f"{stable_identity}@{scheme}#{index}:{hash}"`` where ``index`` is the
    version's position in its lineage (0 == root) and ``hash`` is a short sha256
    over the version's identity-bearing fields (stable identity, resolved node id,
    and lineage index). Two versions of one claim therefore share a stable identity
    but get distinct tags; the same version always hashes to the same tag (no time,
    no randomness, no hash-seed variance), so it is reproducible run-to-run.
    """
    nid = canonical_id(graph, ref)
    if nid is None or nid not in graph.nodes:
        return None
    sid = stable_identity(graph, nid)
    chain = lineage(graph, nid)
    index = chain.index(nid) if nid in chain else 0
    payload = "|".join([sid or "", nid, str(index)])
    return f"{sid}@{_VERSION_SCHEME}#{index}:{_short_hash(payload)}"


# ---------------------------------------------------------------------------
# Round-trip invariant checker
# ---------------------------------------------------------------------------

def _identity_map(graph: Graph) -> "dict[str, dict]":
    """Capture stable_identity + version_tag for every node in a graph."""
    return {
        nid: {"stable": stable_identity(graph, nid), "tag": version_tag(graph, nid)}
        for nid in graph.nodes
    }


def identity_round_trip_report(pages, target: str, *, reason: str = "round-trip probe") -> "dict":
    """Prove identity survives a forget+restore cycle through ``Unlearner``.

    Captures every fact's ``stable_identity`` + ``version_tag`` from the full graph,
    forgets ``target`` (un-grounding its downstream claims), restores it, and
    re-captures. ``stable`` is ``True`` iff no fact's identity drifted across the
    round trip — the "Done when" invariant. ``drifted`` lists any (id, before,
    after) that moved. The verdict carries ``candidateOnly: True`` (no overclaim).
    """
    from agent.unlearning import Unlearner  # noqa: PLC0415

    before = _identity_map(build_graph(list(pages)))

    u = Unlearner(pages)
    forget_res = u.forget(target, reason=reason)
    # Identity must not depend on grounding: capture while target is tombstoned.
    during = _identity_map(u.graph())
    u.restore(target)
    after = _identity_map(build_graph(u.active_pages()))

    drifted: list[dict] = []
    for nid in sorted(before):
        b = before[nid]
        a = after.get(nid)
        if a is None:
            drifted.append({"id": nid, "reason": "missing_after_restore",
                            "before": b, "after": None})
        elif a != b:
            drifted.append({"id": nid, "reason": "identity_drift",
                            "before": b, "after": a})

    # Also assert the surviving (still-present) facts kept identity while tombstoned.
    during_drift: list[str] = []
    for nid, dval in during.items():
        if nid in before and dval != before[nid]:
            during_drift.append(nid)

    return {
        "schema": "sophia.symbol_identity_round_trip.v1",
        "candidateOnly": True,
        "target": target,
        "targetId": forget_res.id,
        "found": forget_res.found,
        "supportLost": list(forget_res.abstain),
        "factsTracked": len(before),
        "stable": not drifted and not during_drift,
        "drifted": drifted,
        "driftedWhileTombstoned": sorted(during_drift),
    }


__all__ = [
    "canonical_id",
    "resolve_all",
    "is_ambiguous",
    "build_sense_index",
    "lineage",
    "current_version",
    "stable_identity",
    "version_tag",
    "identity_round_trip_report",
]
