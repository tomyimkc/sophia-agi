#!/usr/bin/env python3
"""Tests for okf.counterfactual — "what would I conclude if this source were removed?"
plus first-class, auditable retraction.

These exercise the fail-closed semantics: a claim resting only on a removed source
is reported as having LOST its support (not silently kept at face value), while a
claim with an independent ground survives. Dependency-free, offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf.counterfactual import counterfactual_remove, is_grounded, retract  # noqa: E402
from okf.page import Page  # noqa: E402


def _graph():
    # primary -> (derived_solo derives only from primary)
    #         -> (derived_multi derives from primary AND independent)
    # independent is its own ground.
    pages = [
        Page(path=Path("primary.md"), meta={
            "id": "primary", "pageType": "concept",
            "authorConfidence": "consensus", "aliases": ["primary alias"],
        }),
        Page(path=Path("independent.md"), meta={
            "id": "independent", "pageType": "concept", "authorConfidence": "attributed",
        }),
        Page(path=Path("solo.md"), meta={
            "id": "derived_solo", "pageType": "concept",
            "authorConfidence": "consensus", "derivesFrom": ["primary"],
        }),
        Page(path=Path("multi.md"), meta={
            "id": "derived_multi", "pageType": "concept",
            "authorConfidence": "consensus", "derivesFrom": ["primary", "independent"],
        }),
    ]
    return okf.build_graph(pages)


def test_grounding_primary_and_derived() -> None:
    g = _graph()
    assert is_grounded(g, "primary") is True          # root claim grounds itself
    assert is_grounded(g, "derived_solo") is True     # grounded via primary
    # a claim whose only derivesFrom is unresolvable is orphaned -> not grounded
    g2 = okf.build_graph([Page(path=Path("x.md"), meta={
        "id": "orphan", "pageType": "concept", "derivesFrom": ["ghost"]})])
    assert is_grounded(g2, "orphan") is False


def test_counterfactual_removes_solo_support() -> None:
    cf = counterfactual_remove(_graph(), "primary")
    assert cf["found"] is True and cf["id"] == "primary"
    # derived_solo rested ONLY on primary -> support lost
    assert "derived_solo" in cf["supportLost"]
    solo = next(r for r in cf["affected"] if r["page"] == "derived_solo")
    assert solo["supportLost"] is True
    assert solo["confidenceRankAfter"] == 0          # fail-closed collapse


def test_counterfactual_keeps_independently_grounded() -> None:
    cf = counterfactual_remove(_graph(), "primary")
    # derived_multi also derives from independent -> still grounded
    assert "derived_multi" not in cf["supportLost"]
    multi = next(r for r in cf["affected"] if r["page"] == "derived_multi")
    assert multi["dependsOnRemoved"] is True
    assert multi["groundedAfter"] is True
    assert multi["supportLost"] is False


def test_counterfactual_resolves_alias_and_query() -> None:
    cf = counterfactual_remove(_graph(), "primary alias", query="derived_solo")
    assert cf["id"] == "primary"
    q = cf["query"]
    assert q["supportLost"] is True and q["effectiveAfter"] == 0


def test_counterfactual_unknown_source() -> None:
    cf = counterfactual_remove(_graph(), "does-not-exist")
    assert cf["found"] is False and cf["id"] is None


def test_retract_records_reason_and_downstream() -> None:
    r = retract(_graph(), "primary", reason="source shown to be forged", by="curator")
    assert r.found is True and r.id == "primary"
    assert "derived_solo" in r.downstream            # downstream support loss surfaced
    audit = r.audit_entry()
    assert audit["event"] == "retraction"
    assert audit["reason"] == "source shown to be forged"
    assert audit["by"] == "curator"
    assert audit["target"] == "primary"
    assert "derived_solo" in audit["downstream"]


def test_retract_is_non_destructive() -> None:
    g = _graph()
    retract(g, "primary", reason="x")
    # the live graph is unchanged — retraction computes impact, it does not delete
    assert "primary" in g.nodes
    assert okf.belief(g, "derived_solo")["found"] is True


def test_retract_unknown_target() -> None:
    r = retract(_graph(), "ghost", reason="n/a")
    assert r.found is False and r.id is None and r.downstream == []


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_okf_counterfactual: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
