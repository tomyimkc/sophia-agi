#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.symbol_identity — canonical entity + sense layer with versioned ids.

Verifies deterministic resolution (id / alias / ambiguous surface form), the sense
index (context discriminators + ambiguity flag), versioned identity (lineage order,
current head, a stable identity SHARED across a supersedes chain with distinct
per-version tags), version-tag determinism, and the forget+restore identity
invariant. Offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.symbol_identity import (  # noqa: E402
    build_sense_index,
    canonical_id,
    current_version,
    identity_round_trip_report,
    is_ambiguous,
    lineage,
    resolve_all,
    stable_identity,
    version_tag,
)
from okf import build_graph  # noqa: E402
from okf.page import Page  # noqa: E402


def _p(pid, **meta):
    return Page(path=Path(f"{pid}.md"),
                meta={"id": pid, "pageType": "concept", "authorConfidence": "consensus", **meta},
                body=meta.pop("_body", ""))


def _resolution_pages():
    return [
        # A concept with an alias.
        _p("dao_de_jing", aliases=["Tao Te Ching"]),
        # Two figure pages that share the surname "Smith" -> deliberately ambiguous.
        _p("john_smith", pageType="figure", attributedAuthor="John Smith", domain="philosophy"),
        _p("jane_smith", pageType="figure", attributedAuthor="Jane Smith", domain="psychology"),
        # A figure with an unambiguous surname.
        _p("plato_figure", pageType="figure", attributedAuthor="Plato", domain="philosophy"),
    ]


def test_canonical_id_by_id_and_alias() -> None:
    g = build_graph(_resolution_pages())
    # Direct id.
    assert canonical_id(g, "dao_de_jing") == "dao_de_jing"
    # Alias (slugified) resolves to the same page.
    assert canonical_id(g, "Tao Te Ching") == "dao_de_jing"
    assert canonical_id(g, "tao_te_ching") == "dao_de_jing"
    # Wikilink-style target.
    assert canonical_id(g, "[[Dao De Jing]]".strip("[]")) == "dao_de_jing"
    # Unknown ref.
    assert canonical_id(g, "nonexistent_ref") is None
    assert resolve_all(g, "nonexistent_ref") == []


def test_surface_form_resolution_and_ambiguity() -> None:
    g = build_graph(_resolution_pages())
    # Unambiguous surname resolves to the one figure.
    assert canonical_id(g, "Plato") == "plato_figure"
    assert is_ambiguous(g, "Plato") is False
    # Shared surname "Smith" is ambiguous: resolves to BOTH figures, sorted.
    both = resolve_all(g, "Smith")
    assert both == ["jane_smith", "john_smith"]
    assert is_ambiguous(g, "Smith") is True
    # canonical_id still returns a deterministic single pick (sorted-first), not a guess
    # that hides the collision.
    assert canonical_id(g, "Smith") == "jane_smith"


def test_resolve_all_sorted_deterministic() -> None:
    g = build_graph(_resolution_pages())
    # Calling twice yields the same sorted list.
    assert resolve_all(g, "Smith") == resolve_all(g, "Smith") == ["jane_smith", "john_smith"]


def test_sense_index_contexts_and_flags() -> None:
    g_pages = _resolution_pages()
    idx = build_sense_index(g_pages)
    # "smith" is an ambiguous surface form carrying two senses with distinct contexts.
    smith = idx["smith"]
    assert smith["ambiguous"] is True
    ctx_by_id = {s["id"]: s["context"] for s in smith["senses"]}
    assert ctx_by_id["john_smith"] == "figure/philosophy"
    assert ctx_by_id["jane_smith"] == "figure/psychology"
    # senses are sorted by id.
    assert [s["id"] for s in smith["senses"]] == ["jane_smith", "john_smith"]
    # "plato" is unambiguous.
    assert idx["plato"]["ambiguous"] is False
    assert idx["plato"]["senses"][0]["id"] == "plato_figure"
    # alias surface form is indexed.
    assert "tao_te_ching" in idx
    assert idx["tao_te_ching"]["senses"][0]["id"] == "dao_de_jing"


def _supersession_pages():
    # Pluto: planet -> dwarf-planet, a two-version supersession chain.
    return [
        _p("pluto_planet", validUntil="2006", supersededBy=["pluto_dwarf"]),
        _p("pluto_dwarf", validFrom="2006", supersedes=["pluto_planet"]),
        # A longer 3-version chain to test ordering.
        _p("model_v1", supersededBy=["model_v2"]),
        _p("model_v2", supersedes=["model_v1"], supersededBy=["model_v3"]),
        _p("model_v3", supersedes=["model_v2"]),
        # A lone page (1-element lineage).
        _p("lonely_fact"),
    ]


def test_lineage_order_and_current_version() -> None:
    g = build_graph(_supersession_pages())
    # Oldest -> newest regardless of which version we ask from.
    assert lineage(g, "pluto_planet") == ["pluto_planet", "pluto_dwarf"]
    assert lineage(g, "pluto_dwarf") == ["pluto_planet", "pluto_dwarf"]
    assert current_version(g, "pluto_planet") == "pluto_dwarf"
    assert current_version(g, "pluto_dwarf") == "pluto_dwarf"
    # 3-version chain, asked from the middle.
    assert lineage(g, "model_v2") == ["model_v1", "model_v2", "model_v3"]
    assert current_version(g, "model_v1") == "model_v3"
    # Lone page is a 1-element lineage.
    assert lineage(g, "lonely_fact") == ["lonely_fact"]
    assert current_version(g, "lonely_fact") == "lonely_fact"


def test_stable_identity_shared_across_chain_distinct_tags() -> None:
    g = build_graph(_supersession_pages())
    # All versions of the Pluto claim share ONE stable identity.
    sid_old = stable_identity(g, "pluto_planet")
    sid_new = stable_identity(g, "pluto_dwarf")
    assert sid_old is not None
    assert sid_old == sid_new
    # The stable identity is anchored on the lineage root.
    assert sid_old.endswith("pluto_planet")
    # But each version has a DISTINCT version tag.
    tag_old = version_tag(g, "pluto_planet")
    tag_new = version_tag(g, "pluto_dwarf")
    assert tag_old != tag_new
    # Tags carry the shared stable identity as a prefix.
    assert tag_old.startswith(sid_old)
    assert tag_new.startswith(sid_new)
    # Version index encoded: root is #0, successor is #1.
    assert "#0:" in tag_old
    assert "#1:" in tag_new
    # 3-version chain: all share identity, three distinct tags.
    sids = {stable_identity(g, x) for x in ("model_v1", "model_v2", "model_v3")}
    assert len(sids) == 1
    tags = {version_tag(g, x) for x in ("model_v1", "model_v2", "model_v3")}
    assert len(tags) == 3


def test_version_tag_determinism() -> None:
    g = build_graph(_supersession_pages())
    # Reproducible run-to-run: same graph, same tag, twice.
    assert version_tag(g, "pluto_planet") == version_tag(g, "pluto_planet")
    assert version_tag(g, "model_v2") == version_tag(g, "model_v2")
    # And stable across a freshly built graph (no time/randomness in the hash).
    g2 = build_graph(_supersession_pages())
    assert version_tag(g, "pluto_dwarf") == version_tag(g2, "pluto_dwarf")
    assert stable_identity(g, "model_v3") == stable_identity(g2, "model_v3")


def _grounded_chain_pages():
    # A derivesFrom chain: claim derives from a source. Forgetting the source
    # un-grounds the downstream claim; restoring re-grounds it. Identity must not move.
    return [
        _p("witness_source", authorConfidence="attributed"),
        _p("derived_claim", derivesFrom=["witness_source"], authorConfidence="attributed"),
        # A supersession chain that should also keep its shared identity across the trip.
        _p("pluto_planet", supersededBy=["pluto_dwarf"]),
        _p("pluto_dwarf", supersedes=["pluto_planet"]),
    ]


def test_identity_round_trip_stable() -> None:
    pages = _grounded_chain_pages()
    # Sanity: forgetting the source un-grounds derived_claim (so the trip is non-trivial).
    rep = identity_round_trip_report(pages, "witness_source")
    assert rep["found"] is True
    assert rep["candidateOnly"] is True
    assert "derived_claim" in rep["supportLost"]
    # The invariant: no identity drifted across forget+restore.
    assert rep["stable"] is True
    assert rep["drifted"] == []
    assert rep["driftedWhileTombstoned"] == []
    assert rep["factsTracked"] == len(pages)


def test_round_trip_preserves_supersession_identity() -> None:
    pages = _grounded_chain_pages()
    g = build_graph(pages)
    sid_before = stable_identity(g, "pluto_planet")
    rep = identity_round_trip_report(pages, "witness_source")
    assert rep["stable"] is True
    # Rebuild and confirm the shared supersession identity is byte-identical.
    g2 = build_graph(pages)
    assert stable_identity(g2, "pluto_planet") == sid_before
    assert stable_identity(g2, "pluto_dwarf") == sid_before


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
