# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Shadow OKF bulk lattice — quarantined candidate beliefs before boundary promotion.

The *boundary* is the committed OKF graph (wiki pages, provenance-locked). The
*bulk* holds hypothetical nodes and edges that may explore cross-tradition links
or tentative ``derivesFrom`` chains without ever being user-visible until they
pass projection gates. This mirrors ``agent.governed_rsi`` shadow-apply at graph
granularity — not a physics model.

All bulk state is in-memory and non-canonical unless ``okf.projection`` promotes
a node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from okf import wikilinks
from okf.graph import Graph, build, resolve
from okf.page import Page
from okf.schema import as_list


@dataclass
class BulkNode:
    """One candidate page in the bulk lattice."""

    id: str
    meta: dict
    body: str = ""
    bulkOnly: bool = True
    createdAt: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_graph_entry(self) -> dict:
        return {
            "id": self.id,
            "pageType": self.meta.get("pageType", "concept"),
            "meta": dict(self.meta),
            "page": None,
        }


@dataclass
class BulkHypothesis:
    """A tentative structural edge explored only in bulk."""

    source: str
    edge: str
    target: str
    note: str = ""


@dataclass
class BulkGraph:
    """In-memory shadow graph layered over a boundary ``Graph``."""

    boundary: Graph
    nodes: dict[str, BulkNode] = field(default_factory=dict)
    hypotheses: list[BulkHypothesis] = field(default_factory=list)
    relax_tradition: bool = True

    def add_node(self, node_id: str, *, meta: dict, body: str = "") -> BulkNode:
        meta = dict(meta)
        meta.setdefault("id", node_id)
        meta.setdefault("pageType", "concept")
        meta["bulkOnly"] = True
        node = BulkNode(id=node_id, meta=meta, body=body)
        self.nodes[node_id] = node
        return node

    def add_hypothesis(self, source: str, edge: str, target: str, *, note: str = "") -> BulkHypothesis:
        hyp = BulkHypothesis(source=source, edge=edge, target=target, note=note)
        self.hypotheses.append(hyp)
        return hyp

    def combined_graph(self) -> Graph:
        """Merge boundary nodes with bulk nodes and hypothetical edges (for inspection)."""
        pages = []
        for node in self.boundary.nodes.values():
            page = node.get("page")
            if page is not None:
                pages.append(page)
        for bulk in self.nodes.values():
            pages.append(Page(path=f"bulk/{bulk.id}.md", meta=bulk.meta, body=bulk.body))
        graph = build(pages)
        for hyp in self.hypotheses:
            src = resolve(graph, hyp.source)
            if src is None:
                continue
            node = graph.nodes[src]
            key = hyp.edge
            existing = as_list(node["meta"].get(key))
            tgt = wikilinks.normalize_target(hyp.target)
            if tgt not in existing:
                node["meta"][key] = existing + [tgt]
        return graph

    def audit_entry(self) -> dict:
        return {
            "schema": "sophia.okf.bulk_graph.v1",
            "candidateOnly": True,
            "bulkNodeCount": len(self.nodes),
            "hypothesisCount": len(self.hypotheses),
            "relaxTradition": self.relax_tradition,
            "nodeIds": sorted(self.nodes.keys()),
            "hypotheses": [
                {"source": h.source, "edge": h.edge, "target": h.target, "note": h.note}
                for h in self.hypotheses
            ],
        }
