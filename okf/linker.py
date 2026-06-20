"""Link-integrity + contradiction reporting over an OKF wiki.

Builds backlinks, finds dangling links and orphan pages, and folds in the
structural contradiction ledger from okf.graph. Loads the tradition
do-not-merge map from data/traditions.json so tradition-merge detection uses the
same ground truth as the rest of the corpus.
"""

from __future__ import annotations

import json
from pathlib import Path

from okf import graph as okf_graph
from okf import page as okf_page
from okf.schema import as_list

ROOT = Path(__file__).resolve().parents[1]
TRADITIONS_PATH = ROOT / "data" / "traditions.json"

# Page types that legitimately have no inbound links.
_ROOT_TYPES = {"index", "schema", "domain"}


def load_dnm_by_tradition(path: Path = TRADITIONS_PATH) -> dict:
    """tradition id -> [traditions it must not be merged with] (from traditions.json)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict = {}
    for key, record in data.items():
        if isinstance(record, dict):
            dnm = record.get("doNotMergeWith")
            if dnm:
                out[key] = list(dnm)
    return out


def backlinks(graph: okf_graph.Graph) -> dict:
    out: dict = {nid: [] for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        targets = set(okf_graph.out_link_targets(node))
        for key in ("links", "contradicts", "supersedes", "supersededBy", "derivesFrom"):
            targets.update(t for t in (okf_graph._edge_targets(node, key)))
        for target in targets:
            other = okf_graph.resolve(graph, target)
            if other and other != nid and nid not in out[other]:
                out[other].append(nid)
    return out


def orphans(graph: okf_graph.Graph) -> "list[str]":
    back = backlinks(graph)
    out: list[str] = []
    for nid, node in graph.nodes.items():
        if node["pageType"] in _ROOT_TYPES:
            continue
        if not back.get(nid):
            out.append(nid)
    return sorted(out)


def link_report(*roots) -> dict:
    """Full integrity report over every OKF page under the given roots."""
    pages = okf_page.load_pages(*roots)
    graph = okf_graph.build(pages)
    dnm = load_dnm_by_tradition()

    schema_errors: list[dict] = []
    for page in pages:
        errs = page.validate()
        if errs:
            schema_errors.append({"page": page.id, "path": str(page.path), "errors": errs})

    dangling = okf_graph.dangling_links(graph)
    ledger = okf_graph.contradiction_ledger(graph, dnm_by_tradition=dnm)
    orphan_ids = orphans(graph)

    hard = (
        bool(schema_errors)
        or bool(dangling)
        or bool(ledger["selfMerges"])
        or bool(ledger["supersedeCycles"])
        or bool(ledger["confidenceLaundering"])
        or bool(ledger["traditionMerges"])
    )
    return {
        "ok": not hard,
        "pages": len(pages),
        "schemaErrors": schema_errors,
        "danglingLinks": dangling,
        "orphans": orphan_ids,
        "contradictions": ledger,
        "backlinkCount": sum(len(v) for v in backlinks(graph).values()),
    }
