#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""ConsequenceGate tests — the 8th conscience path over the OKF belief graph.

Verifies:
- the cascade = okf.revision.abstain set (we reuse, not re-invent, the primitive);
- flip severity = |abstain|/|graph| and the escalate threshold fires correctly;
- an unresolved retraction target fails closed to ``abstain`` (never a silent no-op);
- the report is non-destructive (the graph is unchanged after a probe);
- wiring into conscience_check: supplying okfGraph routes severe cascades to escalate,
  and NOT supplying it regresses to the 7-path behaviour (empty consequence field);
- 0%-fabrication invariant: the abstain set only ever names *existing* nodes.

Dependency-free, offline, deterministic (no model, no network).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from agent.consequence_gate import (  # noqa: E402
    FLIP_SEVERITY_ESCALATE,
    ConsequenceReport,
    simulate_cascade,
)
from agent.conscience import conscience_check  # noqa: E402
from okf.page import Page  # noqa: E402


def _graph():
    # primary <- mid <- leaf  (transitive derivesFrom chain; same fixture shape as
    # test_okf_revision so the cascade semantics are directly comparable).
    pages = [
        Page(path=Path("p.md"), meta={"id": "primary", "pageType": "concept", "authorConfidence": "consensus"}),
        Page(path=Path("i.md"), meta={"id": "independent", "pageType": "concept", "authorConfidence": "attributed"}),
        Page(path=Path("m.md"), meta={"id": "mid", "pageType": "concept", "derivesFrom": ["primary"]}),
        Page(path=Path("l.md"), meta={"id": "leaf", "pageType": "concept", "derivesFrom": ["mid"]}),
        Page(path=Path("x.md"), meta={"id": "multi", "pageType": "concept", "derivesFrom": ["primary", "independent"]}),
    ]
    return okf.build_graph(pages)


def _big_graph(n_derived: int = 10):
    # primary grounds many leaves, so retracting it flips a large fraction -> escalate.
    pages = [Page(path=Path("p.md"), meta={"id": "primary", "pageType": "concept", "authorConfidence": "consensus"})]
    for i in range(n_derived):
        pages.append(Page(path=Path(f"d{i}.md"), meta={"id": f"d{i}", "pageType": "concept", "derivesFrom": ["primary"]}))
    return okf.build_graph(pages)


def test_cascade_matches_okf_revision_abstain_set() -> None:
    g = _graph()
    rep = simulate_cascade(g, "primary")
    # abstain set = retracted + cascade, exactly okf.revision's contract
    assert set(rep.abstainSet) == {"primary", "mid", "leaf"}
    assert "multi" not in rep.abstainSet  # survives: also grounded in independent


def test_flip_severity_and_escalate_threshold() -> None:
    # 5-node graph, retracting primary -> 3 abstain -> severity 0.6 -> escalate
    g = _graph()
    rep = simulate_cascade(g, "primary")
    assert rep.flipSeverity == round(3 / 5, 4)
    assert rep.flipSeverity >= FLIP_SEVERITY_ESCALATE
    assert rep.verdict == "escalate"
    assert rep.found is True and rep.targetId == "primary"


def test_small_cascade_allows() -> None:
    # retracting `independent` abstains only itself (1/5 = 0.2) — but 0.2 >= 0.15 so
    # it still escalates. To exercise the genuine allow path we need a graph where
    # the retracted node orphans nothing AND is a small fraction. Add an isolated
    # extra ground so the fraction drops below threshold.
    pages = [
        Page(path=Path("p.md"), meta={"id": "primary", "pageType": "concept", "authorConfidence": "consensus"}),
        Page(path=Path("i.md"), meta={"id": "independent", "pageType": "concept", "authorConfidence": "attributed"}),
        Page(path=Path("m.md"), meta={"id": "mid", "pageType": "concept", "derivesFrom": ["primary"]}),
        Page(path=Path("l.md"), meta={"id": "leaf", "pageType": "concept", "derivesFrom": ["mid"]}),
        Page(path=Path("x.md"), meta={"id": "multi", "pageType": "concept", "derivesFrom": ["primary", "independent"]}),
    ]
    # 7 extra isolated grounds -> 12 nodes total; retracting `leaf` abstains 1/12 ~0.083 -> allow
    for k in range(7):
        pages.append(Page(path=Path(f"iso{k}.md"), meta={"id": f"iso{k}", "pageType": "concept", "authorConfidence": "consensus"}))
    g = okf.build_graph(pages)
    rep = simulate_cascade(g, "leaf")
    assert rep.abstainSet == ("leaf",)
    assert rep.verdict == "allow"
    assert rep.flipSeverity < FLIP_SEVERITY_ESCALATE


def test_unresolved_target_fails_closed_to_abstain() -> None:
    g = _graph()
    rep = simulate_cascade(g, "ghost_target_that_does_not_exist")
    assert rep.found is False
    assert rep.targetId is None
    assert rep.verdict == "abstain"  # unbounded consequence -> fail closed
    assert "cannot be bounded" in rep.reason


def test_probe_is_non_destructive() -> None:
    g = _graph()
    before = set(g.nodes)
    simulate_cascade(g, "primary")
    assert set(g.nodes) == before  # the graph is unchanged; we only built a view


def test_abstain_set_only_names_existing_nodes() -> None:
    # 0%-fabrication invariant: the consequence never invents a node id.
    g = _graph()
    rep = simulate_cascade(g, "primary")
    assert set(rep.abstainSet) <= set(g.nodes)


def test_audit_records_retractable_edge() -> None:
    g = _graph()
    rep = simulate_cascade(g, "primary")
    assert len(rep.audit) >= 1
    entry = rep.audit[0]
    assert entry["event"] == "retraction"
    assert entry["target"] == "primary"
    assert "leaf" in entry["cascade"]  # the transitive abstain set is in the record


def test_conscience_wires_consequence_when_graph_supplied() -> None:
    # A neutral text whose consequence would be severe -> kernel routes to escalate.
    # The caller must pass consequenceMove explicitly (the retraction target).
    g = _big_graph(n_derived=10)  # 11 nodes; retracting primary orphans 10 -> 0.909
    dec = conscience_check("neutral probe", context={"okfGraph": g, "consequenceMove": "primary"}).to_dict()
    assert dec["verdict"] == "escalate"
    assert dec["consequence"]["verdict"] == "escalate"
    assert dec["consequence"]["abstainCount"] == 11


def test_conscience_regresses_to_seven_paths_without_graph() -> None:
    # No okfGraph -> consequence field is empty and the 7-path behaviour is intact.
    # Use a known-safe claim so no other path interferes.
    dec = conscience_check("2 + 2 = 4.").to_dict()
    assert dec["verdict"] == "allow"
    assert dec["consequence"] == {}


def test_conscience_skips_consequence_when_move_absent() -> None:
    # FAIL-CLOSED GUARD: a graph is supplied but consequenceMove is missing -> the
    # path MUST skip (empty field) rather than fall back to an arbitrary node. We
    # must never silently retract a real claim from a forgotten key.
    g = _big_graph(n_derived=10)
    dec = conscience_check("2 + 2 = 4.", context={"okfGraph": g}).to_dict()
    assert dec["consequence"] == {}
    assert dec["verdict"] == "allow"  # safe claim, no consequence path ran


def test_conscience_consequence_never_overrides_hard_block() -> None:
    # An AGI-overclaim must STILL block even if a graph + move are supplied and the
    # consequence happens to be benign — hard blocks dominate the 8th path.
    g = _graph()
    dec = conscience_check(
        "Sophia is proven AGI and achieved AGI.",
        context={"okfGraph": g, "consequenceMove": "leaf"},
    ).to_dict()
    assert dec["verdict"] == "block"
    assert dec["constitution"]["verdict"] == "rejected"


def test_consequence_report_schema_and_boundary() -> None:
    g = _graph()
    rep = simulate_cascade(g, "leaf")
    d = rep.to_dict()
    assert d["schema"] == "sophia.consequence.v1"
    assert d["candidateOnly"] is True and d["level3Evidence"] is False
    assert "AGI" in d["boundary"] or "AGI proof" in d["boundary"]


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_consequence_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
