# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Local-global syntactic consistency over OKF belief graphs.

Finds undeclared cross-context disagreements (epistemic holes) when the same
entity carries different asserted claims in different context partitions (e.g.
tradition). Declared ``contradicts`` edges defer to ``contradiction_ledger`` —
no double-report. This checks CONSISTENCY, not truth; it never auto-generates
facts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from okf import wikilinks
from okf.graph import Graph, build, contradiction_ledger, resolve
from okf.schema import as_list

ROOT = Path(__file__).resolve().parents[1]
HOLES_PATH = ROOT / "training" / "feedback" / "epistemic_holes.jsonl"

# Frontmatter fields treated as comparable assertions about an entity.
CLAIM_FIELDS = ("attributedAuthor", "authorConfidence", "domain", "pageType")

# Partition keys pluggable later (source, domain, …).
DEFAULT_PARTITION_KEY = "tradition"

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


def context_of(node: dict, *, partition_key: str = DEFAULT_PARTITION_KEY) -> str:
    """Partition label for one graph node (missing key → ``_none``)."""
    val = node["meta"].get(partition_key)
    return str(val) if val else "_none"


def entity_key_for_page(page) -> str:
    """Stable entity id: canonical title slug when present, else page id."""
    title = page.meta.get("canonicalTitleEn")
    if title:
        return wikilinks.normalize_target(str(title))
    return page.id


def extract_claim(page) -> dict:
    """Assertion bundle carried by a page about its entity."""
    return {k: page.meta[k] for k in CLAIM_FIELDS if page.meta.get(k) is not None}


def claims_overlap_differ(claim_a: dict, claim_b: dict) -> bool:
    """True when shared keys disagree (syntactic, not semantic truth)."""
    shared = set(claim_a) & set(claim_b)
    return any(claim_a[k] != claim_b[k] for k in shared)


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


def _hole_id(entity: str, context_a: str, source_a: str, context_b: str, source_b: str) -> str:
    pair = tuple(sorted([(context_a, source_a), (context_b, source_b)]))
    return f"hole_{entity}_{pair[0][0]}_{pair[0][1]}_{pair[1][0]}_{pair[1][1]}"


def find_epistemic_holes(
    graph: Graph,
    *,
    partition_key: str = DEFAULT_PARTITION_KEY,
) -> list[dict]:
    """Undeclared cross-context disagreements on the same entity."""
    # entity -> context -> {claim, source, sources}
    by_entity: dict[str, dict[str, dict]] = {}

    for nid, node in sorted(graph.nodes.items()):
        page = node["page"]
        entity = entity_key_for_page(page)
        ctx = context_of(node, partition_key=partition_key)
        claim = extract_claim(page)
        if not claim:
            continue
        sources = [str(s) for s in as_list(page.meta.get("sources"))]
        bucket = by_entity.setdefault(entity, {})
        if ctx not in bucket:
            bucket[ctx] = {
                "claim": claim,
                "source": nid,
                "sources": sources,
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
                    "holeId": _hole_id(entity, ctx_a, rec_a["source"], ctx_b, rec_b["source"]),
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


def count_declared_contradictions_deferred(graph: Graph, holes: list[dict]) -> int:
    """Declared contradictions that overlap hole entity pairs (ledger owns these)."""
    ledger = contradiction_ledger(graph)
    declared = ledger.get("declaredContradictions", [])
    deferred = 0
    hole_pairs = {
        tuple(sorted([h["sourceA"], h["sourceB"]]))
        for h in holes
    }
    seen_pairs: set[tuple] = set()
    for row in declared:
        pair = tuple(sorted([row["page"], str(row["contradicts"])]))
        if pair in seen_pairs:
            continue
        if pair in hole_pairs:
            deferred += 1
            seen_pairs.add(pair)
    # Count all declared edges that would have been holes without the edge.
    overlap_count = 0
    for row in declared:
        page_a = row["page"]
        page_b = str(row["contradicts"])
        node_a = graph.nodes.get(page_a)
        node_b = graph.nodes.get(page_b)
        if node_a is None or node_b is None:
            continue
        entity_a = entity_key_for_page(node_a["page"])
        entity_b = entity_key_for_page(node_b["page"])
        if entity_a != entity_b:
            continue
        ctx_a = context_of(node_a)
        ctx_b = context_of(node_b)
        if ctx_a == ctx_b:
            continue
        rec_a = extract_claim(node_a["page"])
        rec_b = extract_claim(node_b["page"])
        if claims_overlap_differ(rec_a, rec_b):
            overlap_count += 1
    return max(overlap_count, deferred)


def consistency_report(
    pages,
    *,
    partition_key: str = DEFAULT_PARTITION_KEY,
    dnm_by_tradition: dict | None = None,
) -> dict:
    """Deterministic summary: contexts, spanning entities, holes, deferred ledger rows."""
    graph = build(pages)
    holes = find_epistemic_holes(graph, partition_key=partition_key)
    contexts = sorted({
        context_of(node, partition_key=partition_key)
        for node in graph.nodes.values()
    })
    entities_spanning = sum(
        1 for entity, ctx_map in _entity_context_map(graph, partition_key).items()
        if len(ctx_map) > 1
    )
    ledger = contradiction_ledger(graph, dnm_by_tradition=dnm_by_tradition)
    declared_n = len(ledger.get("declaredContradictions", []))
    deferred_n = _count_deferred_by_declared_edge(graph, partition_key)

    return {
        "schema": "sophia.okf_consistency_report.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "partitionKey": partition_key,
        "contextCount": len(contexts),
        "contexts": contexts,
        "entitiesSpanningContexts": entities_spanning,
        "epistemicHoleCount": len(holes),
        "declaredContradictionsDeferred": deferred_n,
        "declaredContradictionCount": declared_n,
        "epistemicHoles": holes,
        "boundary": (
            "Syntactic local-global consistency only; escalates disagreements, "
            "does not decide truth or auto-generate training facts."
        ),
    }


def _entity_context_map(graph: Graph, partition_key: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for nid, node in graph.nodes.items():
        entity = entity_key_for_page(node["page"])
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
            entity_a = entity_key_for_page(node["page"])
            entity_b = entity_key_for_page(other["page"])
            if entity_a != entity_b:
                continue
            ctx_a = context_of(node, partition_key=partition_key)
            ctx_b = context_of(other, partition_key=partition_key)
            if ctx_a == ctx_b:
                continue
            if claims_overlap_differ(extract_claim(node["page"]), extract_claim(other["page"])):
                count += 1
    return count


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
    "entity_key_for_page",
    "extract_claim",
    "claims_overlap_differ",
    "has_declared_contradiction",
    "find_epistemic_holes",
    "consistency_report",
    "write_epistemic_holes",
    "propose_hole_patch",
    "_FAIL_CLOSED_VERDICTS",
]
