# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Local-global syntactic consistency over OKF belief graphs.

Finds undeclared cross-page attribution disagreements (epistemic holes) when
>=2 pages assert conflicting attribution about the same referent (work/figure)
via ``links``, ``attributedAuthor``, or ``doNotAttributeTo``. Declared
``contradicts`` / ledger rows defer — no double-report. This checks CONSISTENCY,
not truth; it never auto-generates facts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from okf import wikilinks
from okf.graph import Graph, build, contradiction_ledger, out_link_targets, resolve
from okf.schema import as_list

ROOT = Path(__file__).resolve().parents[1]
HOLES_PATH = ROOT / "training" / "feedback" / "epistemic_holes.jsonl"

# Frontmatter fields treated as comparable assertions about a referent.
CLAIM_FIELDS = ("attributedAuthor", "authorConfidence", "domain", "pageType")

# Default mode: referent-attribution graph (not tradition partition).
DEFAULT_PARTITION_KEY = "referent"

_FAIL_CLOSED_VERDICTS = frozenset({"block", "abstain", "escalate", "retrieve", "clarify"})

_CONSISTENCY_ONLY_PATTERNS = (
    r"\bdeclare\s+(?:these\s+)?consistent\b",
    r"\badd\s+contradicts\b",
    r"\bmerge\s+claims\b",
    r"\balign\s+without\s+(?:new\s+)?(?:source|grounding)\b",
    r"\bmark\s+as\s+consistent\b",
)

_SOURCE_MARKERS = ("data/", "http://", "https://", "wiki/", ".json#", "doi:", "isbn:")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_id(value: str) -> str:
    """Case/slug-normalize referent and author ids."""
    return wikilinks.normalize_target(str(value))


def context_of(node: dict, *, partition_key: str = DEFAULT_PARTITION_KEY) -> str:
    """Partition label for one graph node (missing key → ``_none``). Legacy tradition mode."""
    val = node["meta"].get(partition_key)
    return str(val) if val else "_none"


def referent_key_for_page(page) -> str:
    """Stable referent id: canonical title slug when present, else page id."""
    title = page.meta.get("canonicalTitleEn")
    if title:
        return normalize_id(str(title))
    return normalize_id(page.id)


def entity_key_for_page(page) -> str:
    """Alias for ``referent_key_for_page`` (backward compat)."""
    return referent_key_for_page(page)


def extract_claim(page) -> dict:
    """Assertion bundle carried by a page about its referent."""
    return {k: page.meta[k] for k in CLAIM_FIELDS if page.meta.get(k) is not None}


def claims_overlap_differ(claim_a: dict, claim_b: dict) -> bool:
    """True when shared keys disagree (syntactic, not semantic truth)."""
    shared = set(claim_a) & set(claim_b)
    return any(claim_a[k] != claim_b[k] for k in shared)


def _norm_author(author: object) -> str | None:
    if author is None:
        return None
    s = str(author).strip()
    return normalize_id(s) if s else None


def _dnm_set(meta: dict) -> set[str]:
    return {_norm_author(a) for a in as_list(meta.get("doNotAttributeTo")) if _norm_author(a)}


def referents_for_attribution(graph: Graph, nid: str) -> list[str]:
    """Referents (work/figure) this page asserts attribution about."""
    node = graph.nodes[nid]
    page = node["page"]
    meta = page.meta
    if not meta.get("attributedAuthor") and not as_list(meta.get("doNotAttributeTo")):
        return []

    self_ref = referent_key_for_page(page)
    work_links: list[str] = []
    for target in out_link_targets(node):
        resolved = resolve(graph, target)
        if resolved:
            other = graph.nodes[resolved]
            if other["pageType"] == "tradition":
                continue
            work_links.append(referent_key_for_page(other["page"]))
        else:
            work_links.append(normalize_id(target))

    # Cataloged works assert on themselves; commentary pages assert on linked works.
    if meta.get("canonicalTitleEn"):
        return [self_ref]
    if work_links:
        return sorted(set(work_links))
    return [self_ref]


def attribution_conflict_type(rec_a: dict, rec_b: dict) -> str | None:
    """Return conflict type when two pages disagree on attribution for one referent."""
    auth_a = _norm_author(rec_a.get("attributedAuthor"))
    auth_b = _norm_author(rec_b.get("attributedAuthor"))
    dnm_a = {_norm_author(a) for a in rec_a.get("doNotAttributeTo", []) if _norm_author(a)}
    dnm_b = {_norm_author(a) for a in rec_b.get("doNotAttributeTo", []) if _norm_author(a)}

    if auth_a and auth_b and auth_a != auth_b:
        return "attributed_author_mismatch"
    if auth_a and auth_a in dnm_b:
        return "do_not_attribute_violation"
    if auth_b and auth_b in dnm_a:
        return "do_not_attribute_violation"
    return None


def has_declared_contradiction(graph: Graph, page_a: str, page_b: str) -> bool:
    """Either page declares ``contradicts`` pointing at the other."""
    for src, dst in ((page_a, page_b), (page_b, page_a)):
        node = graph.nodes.get(src)
        if node is None:
            continue
        for raw in as_list(node["meta"].get("contradicts")):
            other = resolve(graph, raw)
            if other == dst:
                return True
    return False


def is_attribution_conflict_deferred(
    graph: Graph,
    page_a: str,
    page_b: str,
    *,
    dnm_by_tradition: dict | None = None,
) -> bool:
    """Declared ``contradicts`` or ledger tradition-merge row defers re-emission."""
    if has_declared_contradiction(graph, page_a, page_b):
        return True
    ledger = contradiction_ledger(graph, dnm_by_tradition=dnm_by_tradition)
    pair = tuple(sorted([page_a, page_b]))
    for row in ledger.get("traditionMerges", []):
        if tuple(sorted([row["page"], row["linksTo"]])) == pair:
            return True
    return False


def _build_referent_assertions(graph: Graph) -> dict[str, dict[str, dict]]:
    """referent -> page_id -> assertion record."""
    by_referent: dict[str, dict[str, dict]] = {}
    for nid, node in sorted(graph.nodes.items()):
        referents = referents_for_attribution(graph, nid)
        if not referents:
            continue
        page = node["page"]
        meta = page.meta
        rec = {
            "source": nid,
            "attributedAuthor": meta.get("attributedAuthor"),
            "doNotAttributeTo": [str(a) for a in as_list(meta.get("doNotAttributeTo"))],
            "claim": extract_claim(page),
            "sources": [str(s) for s in as_list(meta.get("sources"))],
        }
        for ref in referents:
            by_referent.setdefault(ref, {})[nid] = rec
    return by_referent


def _hole_id_referent(referent: str, source_a: str, source_b: str, conflict_type: str) -> str:
    pair = tuple(sorted([source_a, source_b]))
    return f"hole_{referent}_{pair[0]}_{pair[1]}_{conflict_type}"


def _hole_id_tradition(entity: str, context_a: str, source_a: str, context_b: str, source_b: str) -> str:
    pair = tuple(sorted([(context_a, source_a), (context_b, source_b)]))
    return f"hole_{entity}_{pair[0][0]}_{pair[0][1]}_{pair[1][0]}_{pair[1][1]}"


def find_referent_attribution_holes(
    graph: Graph,
    *,
    dnm_by_tradition: dict | None = None,
) -> tuple[list[dict], int]:
    """Undeclared cross-page attribution conflicts on shared referents."""
    by_referent = _build_referent_assertions(graph)
    holes: list[dict] = []
    deferred = 0

    for referent in sorted(by_referent):
        page_map = by_referent[referent]
        if len(page_map) < 2:
            continue
        page_list = sorted(page_map.values(), key=lambda r: r["source"])
        for i, rec_a in enumerate(page_list):
            for rec_b in page_list[i + 1:]:
                conflict_type = attribution_conflict_type(rec_a, rec_b)
                if not conflict_type:
                    continue
                src_a = rec_a["source"]
                src_b = rec_b["source"]
                if is_attribution_conflict_deferred(
                    graph, src_a, src_b, dnm_by_tradition=dnm_by_tradition,
                ):
                    deferred += 1
                    continue
                holes.append({
                    "schema": "sophia.epistemic_hole.v1",
                    "holeId": _hole_id_referent(referent, src_a, src_b, conflict_type),
                    "referent": referent,
                    "entity": referent,
                    "pageA": src_a,
                    "pageB": src_b,
                    "sourceA": src_a,
                    "sourceB": src_b,
                    "claimA": rec_a["claim"],
                    "claimB": rec_b["claim"],
                    "conflictType": conflict_type,
                    "candidateOnly": True,
                    "level3Evidence": False,
                    "resolved": False,
                })
    return holes, deferred


def _find_tradition_partition_holes(
    graph: Graph,
    *,
    partition_key: str = "tradition",
) -> list[dict]:
    """Legacy tradition-partition hole detection."""
    by_entity: dict[str, dict[str, dict]] = {}

    for nid, node in sorted(graph.nodes.items()):
        page = node["page"]
        entity = referent_key_for_page(page)
        ctx = context_of(node, partition_key=partition_key)
        claim = extract_claim(page)
        if not claim:
            continue
        bucket = by_entity.setdefault(entity, {})
        if ctx not in bucket:
            bucket[ctx] = {
                "claim": claim,
                "source": nid,
                "sources": [str(s) for s in as_list(page.meta.get("sources"))],
            }

    holes: list[dict] = []
    for entity in sorted(by_entity):
        ctx_map = by_entity[entity]
        if len(ctx_map) < 2:
            continue
        contexts = sorted(ctx_map)
        for i, ctx_a in enumerate(contexts):
            for ctx_b in contexts[i + 1:]:
                rec_a = ctx_map[ctx_a]
                rec_b = ctx_map[ctx_b]
                if not claims_overlap_differ(rec_a["claim"], rec_b["claim"]):
                    continue
                if has_declared_contradiction(graph, rec_a["source"], rec_b["source"]):
                    continue
                holes.append({
                    "schema": "sophia.epistemic_hole.v1",
                    "holeId": _hole_id_tradition(
                        entity, ctx_a, rec_a["source"], ctx_b, rec_b["source"],
                    ),
                    "entity": entity,
                    "contextA": ctx_a,
                    "claimA": rec_a["claim"],
                    "sourceA": rec_a["source"],
                    "contextB": ctx_b,
                    "claimB": rec_b["claim"],
                    "sourceB": rec_b["source"],
                    "candidateOnly": True,
                    "level3Evidence": False,
                    "resolved": False,
                })
    return holes


def find_epistemic_holes(
    graph: Graph,
    *,
    partition_key: str = DEFAULT_PARTITION_KEY,
    dnm_by_tradition: dict | None = None,
) -> list[dict]:
    """Undeclared disagreements — referent-attribution by default, tradition if requested."""
    if partition_key == "tradition":
        return _find_tradition_partition_holes(graph, partition_key=partition_key)
    holes, _ = find_referent_attribution_holes(graph, dnm_by_tradition=dnm_by_tradition)
    return holes


def count_shared_referents(graph: Graph) -> int:
    """Referents with attribution assertions from >=2 distinct pages."""
    by_referent = _build_referent_assertions(graph)
    return sum(1 for page_map in by_referent.values() if len(page_map) >= 2)


def consistency_report(
    pages,
    *,
    partition_key: str = DEFAULT_PARTITION_KEY,
    dnm_by_tradition: dict | None = None,
) -> dict:
    """Deterministic summary: shared referents, holes, deferred ledger rows."""
    graph = build(pages)
    ledger = contradiction_ledger(graph, dnm_by_tradition=dnm_by_tradition)
    declared_n = len(ledger.get("declaredContradictions", []))

    if partition_key == "tradition":
        holes = _find_tradition_partition_holes(graph, partition_key=partition_key)
        contexts = sorted({
            context_of(node, partition_key=partition_key)
            for node in graph.nodes.values()
        })
        entities_spanning = sum(
            1 for entity, ctx_map in _entity_context_map(graph, partition_key).items()
            if len(ctx_map) > 1
        )
        deferred_n = _count_deferred_by_declared_edge(graph, partition_key)
        shared_referents = 0
    else:
        holes, deferred_n = find_referent_attribution_holes(
            graph, dnm_by_tradition=dnm_by_tradition,
        )
        contexts = []
        entities_spanning = 0
        shared_referents = count_shared_referents(graph)

    gate_pass = len(holes) == 0

    return {
        "schema": "sophia.okf_consistency_report.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "partitionKey": partition_key,
        "contextCount": len(contexts),
        "contexts": contexts,
        "sharedReferents": shared_referents,
        "entitiesSpanningContexts": entities_spanning,
        "epistemicHoleCount": len(holes),
        "declaredContradictionsDeferred": deferred_n,
        "declaredContradictionCount": declared_n,
        "gatePass": gate_pass,
        "epistemicHoles": holes,
        "boundary": (
            "Syntactic local-global consistency only; escalates disagreements, "
            "does not decide truth or auto-generate training facts."
        ),
    }


def _entity_context_map(graph: Graph, partition_key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for nid, node in graph.nodes.items():
        entity = referent_key_for_page(node["page"])
        ctx = context_of(node, partition_key=partition_key)
        out.setdefault(entity, {})[ctx] = nid
    return out


def _count_deferred_by_declared_edge(graph: Graph, partition_key: str) -> int:
    """Pairs with overlapping disagreeing claims AND a declared contradicts edge."""
    count = 0
    for nid, node in sorted(graph.nodes.items()):
        for raw in as_list(node["meta"].get("contradicts")):
            other_id = resolve(graph, raw)
            if other_id is None or other_id not in graph.nodes:
                continue
            other = graph.nodes[other_id]
            entity_a = referent_key_for_page(node["page"])
            entity_b = referent_key_for_page(other["page"])
            if entity_a != entity_b:
                continue
            ctx_a = context_of(node, partition_key=partition_key)
            ctx_b = context_of(other, partition_key=partition_key)
            if ctx_a == ctx_b:
                continue
            if claims_overlap_differ(extract_claim(node["page"]), extract_claim(other["page"])):
                count += 1
    return count


def sync_epistemic_holes(holes: list[dict], *, path: Path | None = None) -> dict:
    """Replace feedback queue with current undeclared holes from a full scan."""
    out = path or HOLES_PATH
    rows: list[dict] = []
    for hole in holes:
        row = dict(hole)
        row.setdefault("schema", "sophia.epistemic_hole.v1")
        row.setdefault("resolved", False)
        row.setdefault("candidateOnly", True)
        row.setdefault("level3Evidence", False)
        rows.append(row)
    _write_jsonl(out, rows)
    return {
        "schema": "sophia.write_epistemic_holes.v1",
        "candidateOnly": True,
        "path": str(out),
        "added": len(rows),
        "total": len(rows),
        "replaced": True,
    }


def write_epistemic_holes(holes: list[dict], *, path: Path | None = None) -> dict:
    """Append new holes to the feedback queue (``resolved: false``). Deduped by holeId."""
    out = path or HOLES_PATH
    existing = _read_jsonl(out)
    seen = {r.get("holeId") for r in existing}
    added = 0
    for hole in holes:
        hid = hole.get("holeId")
        if not hid or hid in seen:
            continue
        row = dict(hole)
        row.setdefault("schema", "sophia.epistemic_hole.v1")
        row.setdefault("resolved", False)
        row.setdefault("candidateOnly", True)
        row.setdefault("level3Evidence", False)
        existing.append(row)
        seen.add(hid)
        added += 1
    _write_jsonl(out, existing)
    return {
        "schema": "sophia.write_epistemic_holes.v1",
        "candidateOnly": True,
        "path": str(out),
        "added": added,
        "total": len(existing),
    }


def _has_real_source(sources: list[str]) -> bool:
    for raw in sources:
        s = str(raw).strip()
        if not s:
            continue
        if any(marker in s for marker in _SOURCE_MARKERS):
            return True
    return False


def _is_consistency_only_without_grounding(text: str, sources: list[str]) -> bool:
    if _has_real_source(sources):
        return False
    low = text.lower()
    return any(re.search(pat, low) for pat in _CONSISTENCY_ONLY_PATTERNS)


def propose_hole_patch(
    hole_id: str,
    resolution_text: str,
    *,
    sources: list[str] | None = None,
    path: Path | None = None,
    skip_conscience: bool = False,
) -> dict:
    """Provenance-gated flywheel patch candidate for one hole (default-deny).

    Requires (a) a real source citation and (b) conscience gate pass. Resolutions
    that only reconcile the graph without grounding are rejected.
    """
    from agent.conscience import conscience_check

    srcs = [str(s) for s in (sources or [])]
    if not _has_real_source(srcs):
        return {
            "ok": False,
            "rejected": True,
            "defaultDeny": True,
            "holeId": hole_id,
            "reason": "no provenance source citation",
        }
    if _is_consistency_only_without_grounding(resolution_text, srcs):
        return {
            "ok": False,
            "rejected": True,
            "defaultDeny": True,
            "holeId": hole_id,
            "reason": "consistency-only resolution without new grounding",
        }

    if not skip_conscience:
        decision = conscience_check(
            resolution_text,
            mode="output",
            action="draft_output",
            context={"canClaimAGI": False},
        )
        if decision.verdict in _FAIL_CLOSED_VERDICTS:
            return {
                "ok": False,
                "rejected": True,
                "defaultDeny": True,
                "holeId": hole_id,
                "reason": f"conscience {decision.verdict}: {decision.reason}",
                "conscience": decision.to_dict(),
            }

    pending = path or HOLES_PATH
    rows = _read_jsonl(pending)
    target: dict | None = None
    for row in rows:
        if row.get("holeId") == hole_id:
            target = row
            break
    if target is None:
        return {"ok": False, "error": f"no hole '{hole_id}' in queue", "defaultDeny": True}

    target["patchCandidate"] = True
    target["resolutionText"] = resolution_text
    target["resolutionSources"] = srcs
    target["patchGate"] = "passed"
    _write_jsonl(pending, rows)
    return {
        "ok": True,
        "holeId": hole_id,
        "patchCandidate": True,
        "candidateOnly": True,
        "defaultDeny": False,
    }


__all__ = [
    "HOLES_PATH",
    "DEFAULT_PARTITION_KEY",
    "CLAIM_FIELDS",
    "context_of",
    "referent_key_for_page",
    "entity_key_for_page",
    "normalize_id",
    "extract_claim",
    "claims_overlap_differ",
    "referents_for_attribution",
    "attribution_conflict_type",
    "has_declared_contradiction",
    "is_attribution_conflict_deferred",
    "find_referent_attribution_holes",
    "find_epistemic_holes",
    "count_shared_referents",
    "consistency_report",
    "sync_epistemic_holes",
    "write_epistemic_holes",
    "propose_hole_patch",
    "_FAIL_CLOSED_VERDICTS",
]
