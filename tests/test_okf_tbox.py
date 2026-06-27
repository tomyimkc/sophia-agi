#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Graph-level concept-TBox consistency checks (P3). Offline.

Exercises the detectors wired into okf.graph.contradiction_ledger:
  - subclassCycles                 (X ⊑ Y ⊑ X)
  - disjointnessViolations         (X ⊑ C1 and X ⊑ C2, C1 disjointWith C2)
  - unsupportedOntologyEdges       (TBox edge resting on no provenance)
  - crossTraditionUnscopedMappings (cross-tradition subClassOf / unscoped analogy)

See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402
from okf import graph as okf_graph  # noqa: E402
from okf import page as okf_page  # noqa: E402


def _write(dir_: Path, rel: str, meta: dict, body: str = "body") -> None:
    path = dir_ / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.serialize(meta, body), encoding="utf-8")


def test_subclass_cycle_detected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "x.md", {"id": "x", "pageType": "concept", "subClassOf": ["y"], "sources": ["s"]})
        _write(d, "y.md", {"id": "y", "pageType": "concept", "subClassOf": ["x"], "sources": ["s"]})
        ledger = okf_graph.contradiction_ledger(okf_graph.build(okf_page.load_pages(d)))
        assert ledger["subclassCycles"], "expected a subClassOf cycle x<->y"


def test_disjointness_violation_detected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "c1.md", {"id": "c1", "pageType": "concept", "disjointWith": ["c2"], "sources": ["s"]})
        _write(d, "c2.md", {"id": "c2", "pageType": "concept", "sources": ["s"]})
        _write(d, "child.md", {"id": "child", "pageType": "concept", "subClassOf": ["c1", "c2"], "sources": ["s"]})
        ledger = okf_graph.contradiction_ledger(okf_graph.build(okf_page.load_pages(d)))
        pages = {v["page"] for v in ledger["disjointnessViolations"]}
        assert "child" in pages, ledger["disjointnessViolations"]


def test_unsupported_ontology_edge_flagged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # subClassOf edge with no sources and no confidence -> unsupported.
        _write(d, "weak.md", {"id": "weak", "pageType": "concept", "subClassOf": ["base"]})
        _write(d, "base.md", {"id": "base", "pageType": "concept", "sources": ["s"]})
        ledger = okf_graph.contradiction_ledger(okf_graph.build(okf_page.load_pages(d)))
        pages = {v["page"] for v in ledger["unsupportedOntologyEdges"]}
        assert "weak" in pages, ledger["unsupportedOntologyEdges"]
        assert "base" not in pages


def test_cross_tradition_mapping_scope_required() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "wuwei.md", {"id": "wuwei", "pageType": "concept", "tradition": "daoist",
                               "scopedAnalogy": ["apatheia"], "sources": ["s"]})  # unscoped -> flagged
        _write(d, "ren.md", {"id": "ren", "pageType": "concept", "tradition": "confucian",
                             "subClassOf": ["agape"], "sources": ["s"]})  # cross-tradition subClassOf -> flagged
        _write(d, "apatheia.md", {"id": "apatheia", "pageType": "concept", "tradition": "stoic", "sources": ["s"]})
        _write(d, "agape.md", {"id": "agape", "pageType": "concept", "tradition": "christianity", "sources": ["s"]})
        _write(d, "scoped.md", {"id": "scoped", "pageType": "concept", "tradition": "daoist",
                                "scopedAnalogy": ["apatheia"], "analogyScope": "effortless non-attached response",
                                "sources": ["s"]})  # scoped -> NOT flagged
        ledger = okf_graph.contradiction_ledger(okf_graph.build(okf_page.load_pages(d)))
        flagged = {v["page"] for v in ledger["crossTraditionUnscopedMappings"]}
        assert "wuwei" in flagged
        assert "ren" in flagged
        assert "scoped" not in flagged, ledger["crossTraditionUnscopedMappings"]


def main() -> int:
    test_subclass_cycle_detected()
    test_disjointness_violation_detected()
    test_unsupported_ontology_edge_flagged()
    test_cross_tradition_mapping_scope_required()
    print("test_okf_tbox: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
