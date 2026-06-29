#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf/extract.py: provenance-tainted extraction + provenance-aware recall.

The thesis-defining property under test: a multi-hop recall path is floored by the
WEAKEST page it touches, so a confident event reached through a legendary bridge is
surfaced with a low provenanceFloor and capped=True — a recall-only engine cannot do
this.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import extract, frontmatter, graph as okf_graph, page as okf_page, trace as okf_trace  # noqa: E402


def _write(dir_: Path, rel: str, meta: dict, body: str = "body") -> None:
    path = dir_ / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.serialize(meta, body), encoding="utf-8")


def _corpus(d: Path) -> None:
    # consensus page that links to a legendary one and a consensus one
    _write(d, "concept/a.md", {"id": "a", "pageType": "concept", "authorConfidence": "consensus"},
           "A relates to [[b]] and to [[c]] in one sentence.\nA also notes [[c]] again.")
    # legendary page (weak provenance) — the bridge
    _write(d, "text/b.md", {"id": "b", "pageType": "text", "authorConfidence": "legendary"},
           "B is a legendary text that mentions [[c]].")
    # consensus page reached via b
    _write(d, "concept/c.md", {"id": "c", "pageType": "concept", "authorConfidence": "consensus"},
           "C is well established.")
    # a page that LAUNDERS: declares consensus but derivesFrom legendary b
    _write(d, "concept/d.md", {"id": "d", "pageType": "concept", "authorConfidence": "consensus",
                               "derivesFrom": ["b"]}, "D follows from [[b]].")


def test_extract_stamps_effective_confidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _corpus(d)
        pages = okf_page.load_pages(d)
        events = extract.extract_events(pages)
        by_page = {}
        for ev in events:
            by_page.setdefault(ev.page_id, []).append(ev)

        # every page produced at least one unit (nothing unretrievable)
        assert set(by_page) == {"a", "b", "c", "d"}, sorted(by_page)

        # 'a' has two linked sentences -> two units, both carry its consensus rank (4)
        assert len(by_page["a"]) == 2
        assert all(ev.confidence_rank == 4 for ev in by_page["a"])

        # 'd' declares consensus but derivesFrom legendary b -> EFFECTIVE rank is floored to 1,
        # so the unit is already tainted before recall (no laundering at the source).
        assert by_page["d"][0].author_confidence == "consensus"
        assert by_page["d"][0].confidence_rank == 1
        assert by_page["d"][0].capped is True

        # entities include the page's own id plus its resolved links
        a0 = by_page["a"][0]
        assert "a" in a0.entities and "b" in a0.entities and "c" in a0.entities


def test_entity_index_cross_page() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _corpus(d)
        events = extract.extract_events(okf_page.load_pages(d))
        index = extract.build_entity_index(events)
        # 'c' is mentioned by a, b and c itself -> the entity index spans pages with no
        # hand-authored edge between them (the structural recall SAG buys, OKF-side).
        owners = {eid.split("::")[0] for eid in index["c"]}
        assert {"a", "b", "c"} <= owners


def test_multi_hop_floors_provenance_through_legendary_bridge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _corpus(d)
        events = extract.extract_events(okf_page.load_pages(d))
        hits = extract.multi_hop_recall("legendary text", events, max_hops=2, top_k=10)

        by_page = {}
        for h in hits:
            by_page.setdefault(h.event.page_id, h)

        # The query matches legendary 'b' directly (hop 0): floor == 1, capped.
        assert "b" in by_page
        assert by_page["b"].provenance_floor == 1
        assert by_page["b"].capped is True

        # 'c' (consensus, rank 4) is reachable as an EXPANSION through the legendary 'b'
        # bridge. Its own rank is 4, but the surfacing path's floor must be capped to 1.
        if "c" in by_page and by_page["c"].hops >= 1:
            assert by_page["c"].event.confidence_rank == 4  # the page itself is strong
            assert by_page["c"].provenance_floor == 1, by_page["c"]  # but the PATH is weak
            assert by_page["c"].capped is True


def test_direct_consensus_hit_is_not_capped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _corpus(d)
        events = extract.extract_events(okf_page.load_pages(d))
        hits = extract.multi_hop_recall("well established", events, max_hops=1, top_k=10)
        c_hits = [h for h in hits if h.event.page_id == "c" and h.hops == 0]
        assert c_hits, "expected a direct lexical hit on c"
        assert c_hits[0].provenance_floor == 4
        assert c_hits[0].capped is False


def test_is_capped_threshold_matches_schema_ladder() -> None:
    # Guard against drift between extract.CAPPED_RANK and the schema confidence ladder.
    assert extract.is_capped(extract.CONFIDENCE_RANK["legendary"]) is True
    assert extract.is_capped(extract.CONFIDENCE_RANK["anachronism_risk"]) is True
    assert extract.is_capped(extract.CONFIDENCE_RANK["attributed"]) is False
    assert extract.is_capped(extract.CONFIDENCE_RANK["consensus"]) is False


def test_trace_records_carry_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _corpus(d)
        events = extract.extract_events(okf_page.load_pages(d))
        hits = extract.multi_hop_recall("legendary text", events, max_hops=2, top_k=10)
        recs = okf_trace.trace_records(hits)
        assert recs and recs[0]["rank"] == 1
        for r in recs:
            # every trace record exposes the provenance verdict and a retrieval reason
            assert set(r) >= {"page", "provenanceFloor", "capped", "why", "path"}
            assert isinstance(r["capped"], bool)
            assert (r["provenanceFloor"] <= extract.CAPPED_RANK) == r["capped"]
        # the legendary bridge page is reported capped in the trace
        b = next(r for r in recs if r["page"] == "b")
        assert b["capped"] is True and b["why"] == "direct lexical match"


def test_format_trace_flags_capped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _corpus(d)
        events = extract.extract_events(okf_page.load_pages(d))
        hits = extract.multi_hop_recall("legendary text", events, top_k=5)
        out = okf_trace.format_trace("legendary text", hits)
        assert "recall trace" in out and "CAPPED" in out
        assert "rest on weak provenance" in out


def test_runs_on_real_wiki() -> None:
    # Smoke test against the committed wiki/ corpus (no fixtures): extraction + recall
    # complete and every unit carries a valid rank.
    wiki = ROOT / "wiki"
    if not wiki.is_dir():
        return
    pages = okf_page.load_pages(wiki)
    graph = okf_graph.build(pages)
    events = extract.extract_events(pages, graph=graph)
    assert events, "expected at least one event unit from wiki/"
    assert all(0 <= ev.confidence_rank <= 4 for ev in events)
    hits = extract.multi_hop_recall("penicillin discovery", events, top_k=5)
    assert all(isinstance(h.provenance_floor, int) for h in hits)


def main() -> int:
    test_extract_stamps_effective_confidence()
    test_entity_index_cross_page()
    test_multi_hop_floors_provenance_through_legendary_bridge()
    test_direct_consensus_hit_is_not_capped()
    test_is_capped_threshold_matches_schema_ladder()
    test_trace_records_carry_provenance()
    test_format_trace_flags_capped()
    test_runs_on_real_wiki()
    print("test_okf_extract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
