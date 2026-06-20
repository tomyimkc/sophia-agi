#!/usr/bin/env python3
"""Tests for the okf/ package: frontmatter codec, wikilinks, schema, graph, linker."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf import frontmatter, graph as okf_graph, linker, page as okf_page, schema, wikilinks  # noqa: E402


def test_frontmatter_roundtrip() -> None:
    meta = {
        "id": "dao_de_jing",
        "pageType": "text",
        "tradition": "daoist",
        "authorConfidence": "legendary",
        "canonicalTitleZh": "道德經",
        "canonicalTitleEn": "Dao De Jing",
        "doNotAttributeTo": ["confucius", "socrates", "plato"],
        "doNotMergeWith": [],
        "confidence": 0.4,
        "active": True,
        "retired": False,
        "notes": None,
    }
    body = "# Title\n\nProse with [[a_link]]."
    doc = frontmatter.serialize(meta, body)
    m2, b2 = frontmatter.parse(doc)
    assert m2 == meta, (m2, meta)
    assert b2.strip() == body.strip()


def test_strip_and_no_frontmatter() -> None:
    assert frontmatter.parse("no frontmatter here") == ({}, "no frontmatter here")
    assert frontmatter.strip("---\nid: x\npageType: text\n---\n\nbody") == "body"


def test_quoting_special_values() -> None:
    meta = {"id": "x", "pageType": "text", "title": "A: B, C [need quoting]", "num_like": "007"}
    m2, _ = frontmatter.parse(frontmatter.serialize(meta, "b"))
    assert m2["title"] == "A: B, C [need quoting]"
    assert m2["num_like"] == "007"  # quoted so it stays a string, not int 7


def test_wikilinks() -> None:
    body = "See [[Dao De Jing]], [[republic#book1]] and [[analects|the Analects]]; dup [[republic]]."
    assert wikilinks.extract_links(body) == ["dao_de_jing", "republic", "analects"]
    assert wikilinks.normalize_target("Marco-Polo Pasta") == "marco_polo_pasta"


def test_schema_validate() -> None:
    assert schema.validate_meta({"id": "ok_id", "pageType": "text"}) == []
    bad = schema.validate_meta({"id": "Bad Id", "pageType": "nope", "attributedAuthor": "x"})
    assert any("slug" in e for e in bad)
    assert any("pageType" in e for e in bad)
    assert any("authorConfidence" in e for e in bad)


def _write(dir_: Path, rel: str, meta: dict, body: str = "body") -> None:
    path = dir_ / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.serialize(meta, body), encoding="utf-8")


def test_graph_contradiction_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "text/a.md", {"id": "a", "pageType": "text", "attributedAuthor": "x",
                                 "authorConfidence": "attributed", "doNotAttributeTo": ["x"]})
        _write(d, "text/b.md", {"id": "b", "pageType": "text", "supersedes": ["c"]})
        _write(d, "text/c.md", {"id": "c", "pageType": "text", "supersedes": ["b"]})
        _write(d, "text/e.md", {"id": "e", "pageType": "concept", "authorConfidence": "consensus",
                                "derivesFrom": ["f"]})
        _write(d, "text/f.md", {"id": "f", "pageType": "text", "authorConfidence": "legendary"})
        _write(d, "text/g.md", {"id": "g", "pageType": "text"}, "links to [[does_not_exist]]")

        graph = okf_graph.build(okf_page.load_pages(d))
        ledger = okf_graph.contradiction_ledger(graph)
        assert [s["page"] for s in ledger["selfMerges"]] == ["a"]
        assert ledger["supersedeCycles"], "expected a supersede cycle b<->c"
        assert [c["page"] for c in ledger["confidenceLaundering"]] == ["e"]
        assert {d_["target"] for d_ in okf_graph.dangling_links(graph)} == {"does_not_exist"}


def test_linker_report_clean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write(d, "tradition/daoist.md", {"id": "daoist", "pageType": "tradition"})
        _write(d, "text/dao_de_jing.md", {"id": "dao_de_jing", "pageType": "text",
                                          "tradition": "daoist", "links": ["daoist"]}, "see [[daoist]]")
        rep = linker.link_report(d)
        assert rep["ok"] is True, rep
        assert rep["pages"] == 2
        assert rep["backlinkCount"] >= 1


def main() -> int:
    test_frontmatter_roundtrip()
    test_strip_and_no_frontmatter()
    test_quoting_special_values()
    test_wikilinks()
    test_schema_validate()
    test_graph_contradiction_detection()
    test_linker_report_clean()
    print("test_okf: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
