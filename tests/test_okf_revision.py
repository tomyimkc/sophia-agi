#!/usr/bin/env python3
"""Tests for okf.revision — apply retractions and propagate the support cascade.

Verifies transitive cascade (A grounds B grounds C), multi-retraction, the
fail-closed notFound behaviour, the abstain set a gate would consult, and the
audit log. Dependency-free, offline, deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf.revision import claims_to_abstain, revise  # noqa: E402
from okf.page import Page  # noqa: E402


def _graph():
    # primary <- mid <- leaf  (a transitive derivesFrom chain)
    # independent is its own ground; multi derives from BOTH primary and independent
    pages = [
        Page(path=Path("p.md"), meta={"id": "primary", "pageType": "concept", "authorConfidence": "consensus"}),
        Page(path=Path("i.md"), meta={"id": "independent", "pageType": "concept", "authorConfidence": "attributed"}),
        Page(path=Path("m.md"), meta={"id": "mid", "pageType": "concept", "derivesFrom": ["primary"]}),
        Page(path=Path("l.md"), meta={"id": "leaf", "pageType": "concept", "derivesFrom": ["mid"]}),
        Page(path=Path("x.md"), meta={"id": "multi", "pageType": "concept", "derivesFrom": ["primary", "independent"]}),
    ]
    return okf.build_graph(pages)


def test_cascade_is_transitive() -> None:
    rev = revise(_graph(), ["primary"])
    pages = {c["page"] for c in rev.cascade}
    assert rev.retracted == ["primary"]
    assert "mid" in pages and "leaf" in pages       # mid orphaned, leaf orphaned via mid
    assert "multi" not in pages                       # survives: also grounded in independent


def test_abstain_includes_retracted_and_cascade() -> None:
    ab = claims_to_abstain(_graph(), ["primary"])
    assert "primary" in ab and "mid" in ab and "leaf" in ab
    assert "multi" not in ab and "independent" not in ab


def test_multi_retraction_orphans_multi() -> None:
    # retract BOTH grounds of `multi` -> it loses support too
    rev = revise(_graph(), ["primary", "independent"])
    pages = {c["page"] for c in rev.cascade}
    assert {"mid", "leaf", "multi"} <= pages


def test_not_found_is_reported_not_silent() -> None:
    rev = revise(_graph(), [("ghost", "n/a"), "primary"])
    assert "ghost" in rev.notFound
    assert rev.retracted == ["primary"]


def test_reason_and_audit_log() -> None:
    rev = revise(_graph(), [("primary", "shown to be forged")], by="curator")
    assert rev.reasons["primary"] == "shown to be forged"
    log = rev.audit_log()
    assert len(log) == 1
    assert log[0]["event"] == "retraction" and log[0]["by"] == "curator"
    assert "leaf" in log[0]["cascade"]


def test_confidence_collapses_failclosed() -> None:
    rev = revise(_graph(), ["primary"])
    mid = next(c for c in rev.cascade if c["page"] == "mid")
    assert mid["confidenceRankAfter"] == 0


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_okf_revision: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
