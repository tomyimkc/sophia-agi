# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf.evidence_edges: HARD CONSTRAINTS + determinism on synthetics.

The load-bearing assertions: a proposed edge NEVER crosses a doNotMergeWith, and
a same-lineage/merge-flavoured edge (supports/refines/sameTradition) is NEVER
emitted touching a PROTECTED domain (religion/history) — those pairs may only get
'relatedTo'. Also: mining is deterministic and the score is signal-monotone.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name, rel_path):
    """Load a module by file path, bypassing okf/__init__ (which may drag in a
    Python-3.11+ regex sibling that fails to compile under older runtimes)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence_edges = _load_module("okf_evidence_edges_test", "okf/evidence_edges.py")
MERGE_FLAVOURED = evidence_edges.MERGE_FLAVOURED
PROTECTED_DOMAINS = evidence_edges.PROTECTED_DOMAINS
mine_edges = evidence_edges.mine_edges
score_edge = evidence_edges.score_edge


class _FakePage:
    """A minimal stand-in for okf.Page: has .id and .meta."""

    def __init__(self, pid, meta):
        self._id = pid
        self.meta = dict(meta)

    @property
    def id(self):
        return self._id


def _synthetic_pages():
    return [
        # Two psychology cognitive concepts sharing domain + subfield + a source.
        _FakePage("confirmation_bias", {
            "pageType": "concept", "domain": "psychology", "subfield": "cognitive",
            "canonicalTitleEn": "confirmation bias",
            "sources": ["data/psychology_concepts.json#shared"],
            "doNotMergeWith": [],
        }),
        _FakePage("cognitive_dissonance", {
            "pageType": "concept", "domain": "psychology", "subfield": "cognitive",
            "canonicalTitleEn": "cognitive dissonance",
            "sources": ["data/psychology_concepts.json#shared"],
            "doNotMergeWith": ["freudian"],
        }),
        # A do-not-merge PAIR: evolution must NOT link to lamarckism.
        _FakePage("evolution_natural_selection", {
            "pageType": "concept", "domain": "science", "subfield": "biology",
            "canonicalTitleEn": "evolution by natural selection",
            "attributedAuthor": "charles_darwin",
            "doNotMergeWith": ["lamarckism"],
        }),
        _FakePage("lamarckism", {
            "pageType": "concept", "domain": "science", "subfield": "biology",
            "canonicalTitleEn": "lamarckism inheritance",
            "attributedAuthor": "lamarck",
            "doNotMergeWith": [],
        }),
        # Two religion concepts in the SAME tradition — protected domain, so the
        # strongest kind (sameTradition) must be DEMOTED to relatedTo.
        _FakePage("confucian_ancestor_veneration", {
            "pageType": "concept", "domain": "religion", "tradition": "confucian_ritual",
            "canonicalTitleEn": "ancestor veneration",
            "doNotMergeWith": ["christianity", "islam", "buddhism"],
        }),
        _FakePage("confucian_ritual_reform", {
            "pageType": "concept", "domain": "religion", "tradition": "confucian_ritual",
            "canonicalTitleEn": "ritual reform",
            "doNotMergeWith": [],
        }),
        # A history page sharing a source with a religion page (protected + protected).
        _FakePage("ming_ancestral_rites", {
            "pageType": "event", "domain": "history",
            "canonicalTitleEn": "ming ancestral rites",
            "sources": ["data/history.json#rites"],
            "doNotMergeWith": [],
        }),
        _FakePage("qing_ancestral_rites", {
            "pageType": "event", "domain": "history",
            "canonicalTitleEn": "qing ancestral rites",
            "sources": ["data/history.json#rites"],
            "doNotMergeWith": [],
        }),
        # A page whose doNotMergeWith names the OTHER page's DOMAIN.
        _FakePage("christianity_hub", {
            "pageType": "tradition", "domain": "religion", "tradition": "christianity",
            "canonicalTitleEn": "christianity",
            "doNotMergeWith": ["confucian_ritual"],
        }),
    ]


def test_never_links_across_do_not_merge():
    pages = _synthetic_pages()
    edges = mine_edges(pages)
    # evolution <-> lamarckism is a declared doNotMergeWith pair.
    for e in edges:
        pair = {e["src"], e["dst"]}
        assert pair != {"evolution_natural_selection", "lamarckism"}, \
            f"emitted a doNotMergeWith-crossing edge: {e}"
    # christianity_hub declares doNotMergeWith [confucian_ritual]; it must not
    # link to either confucian page (matched via their tradition).
    for e in edges:
        pair = {e["src"], e["dst"]}
        assert not (
            "christianity_hub" in pair
            and pair & {"confucian_ancestor_veneration", "confucian_ritual_reform"}
        ), f"emitted an edge across a tradition-level doNotMergeWith: {e}"


def test_protected_domain_never_gets_merge_flavoured_edge():
    pages = _synthetic_pages()
    edges = mine_edges(pages)
    domain = {p.id: p.meta.get("domain") for p in pages}
    for e in edges:
        touches_protected = (
            domain.get(e["src"]) in PROTECTED_DOMAINS
            or domain.get(e["dst"]) in PROTECTED_DOMAINS
        )
        if touches_protected:
            assert e["kind"] not in MERGE_FLAVOURED, \
                f"merge-flavoured edge on protected domain: {e}"
            assert e["kind"] == "relatedTo", \
                f"protected pair got non-relatedTo kind: {e}"


def test_same_tradition_protected_is_demoted_but_present():
    """The two same-tradition religion concepts still get an edge — as relatedTo."""
    pages = _synthetic_pages()
    edges = mine_edges(pages)
    found = None
    for e in edges:
        if {e["src"], e["dst"]} == {"confucian_ancestor_veneration", "confucian_ritual_reform"}:
            found = e
    assert found is not None, "expected an edge between same-tradition religion concepts"
    assert found["kind"] == "relatedTo", f"expected demotion to relatedTo, got {found}"
    assert "shared_tradition" in found["evidence"]


def test_non_protected_same_tradition_keeps_kind():
    """A same-tradition pair in a NON-protected domain keeps 'sameTradition'."""
    pages = [
        _FakePage("stoic_a", {"pageType": "concept", "domain": "philosophy",
                              "tradition": "stoic", "canonicalTitleEn": "apatheia"}),
        _FakePage("stoic_b", {"pageType": "concept", "domain": "philosophy",
                              "tradition": "stoic", "canonicalTitleEn": "prohairesis"}),
    ]
    edges = mine_edges(pages)
    assert len(edges) == 1
    assert edges[0]["kind"] == "sameTradition", edges[0]


def test_shared_source_non_protected_is_supports():
    pages = [
        _FakePage("sci_a", {"pageType": "concept", "domain": "science",
                            "canonicalTitleEn": "alpha decay",
                            "sources": ["data/science.json#nuclear"]}),
        _FakePage("sci_b", {"pageType": "concept", "domain": "science",
                            "canonicalTitleEn": "beta decay",
                            "sources": ["data/science.json#nuclear"]}),
    ]
    edges = mine_edges(pages)
    assert len(edges) == 1
    assert edges[0]["kind"] == "supports", edges[0]
    assert "shared_source" in edges[0]["evidence"]


def test_no_merge_or_sameas_kind_ever():
    pages = _synthetic_pages()
    edges = mine_edges(pages)
    for e in edges:
        assert e["kind"] not in ("merge", "sameAs"), e
        assert e["kind"] in evidence_edges.RELATION_KINDS, e


def test_determinism():
    pages = _synthetic_pages()
    a = mine_edges(pages)
    # Reverse input order — output must be identical (canonicalised + sorted).
    b = mine_edges(list(reversed(pages)))
    assert a == b, "mining is not deterministic under input reordering"


def test_score_is_signal_monotone_and_bounded():
    assert score_edge([]) == 0.0
    s1 = score_edge(["shared_domain"])
    s2 = score_edge(["shared_domain", "shared_source"])
    assert 0.0 < s1 < s2 <= 1.0
    # Saturates at 1.0.
    big = score_edge(list(evidence_edges._SIGNAL_WEIGHTS.keys()))
    assert big == 1.0


def test_no_self_edges_and_min_score_filter():
    pages = _synthetic_pages()
    edges = mine_edges(pages, min_score=0.9)
    for e in edges:
        assert e["src"] != e["dst"]
        assert e["score"] >= 0.9


def test_attributions_relationship_signal():
    pages = [
        _FakePage("dao_de_jing", {"pageType": "text", "domain": "philosophy",
                                  "canonicalTitleEn": "dao de jing"}),
        _FakePage("zhuangzi", {"pageType": "text", "domain": "philosophy",
                               "canonicalTitleEn": "zhuangzi"}),
    ]
    attributions = {
        "dao_de_jing": {"tradition": "daoist", "attributedAuthor": "laozi"},
        "zhuangzi": {"tradition": "daoist", "attributedAuthor": "zhuangzi"},
    }
    edges = mine_edges(pages, attributions=attributions)
    assert len(edges) == 1
    assert "attribution_relationship" in edges[0]["evidence"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"ALL TESTS PASSED ({len(tests)} tests)")
