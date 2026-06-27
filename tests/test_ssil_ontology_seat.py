#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SSIL ontology seat (P4) — veto-only concept-edge gate, fail-closed. Offline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil import SSILCandidate, run_ssil  # noqa: E402
from agent.ssil_ontology_seat import g_ontology_gate  # noqa: E402
from agent.continual_plasticity import EvalMetric, UpdateCandidate  # noqa: E402
from agent.corrigibility_gate import CorrigibilityCandidate, load_frozen_eval  # noqa: E402
from agent.honeypot_gate import load_honeypots, select_active  # noqa: E402
from agent.reward_isolation import AccessTrace  # noqa: E402
from agent.ssil_proposer import _full_surface  # noqa: E402


def _edge(subject, obj, etype, ta, tb, **kw):
    e = {"subject": subject, "object": obj, "edgeType": etype, "subjectTradition": ta, "objectTradition": tb}
    e.update(kw)
    return e


def test_seat_vetoes_violation() -> None:
    d = g_ontology_gate([_edge("ancestor_veneration", "eucharist", "scopedAnalogy",
                               "confucian_ritual", "christianity", sources=["x"], scope="ritual")])
    assert d["verdict"] == "reject"


def test_seat_quarantines_unverifiable_identity() -> None:
    d = g_ontology_gate([_edge("ren", "agape", "sameAs", "confucian", "christianity", sources=["x"], scope="love")])
    assert d["verdict"] == "quarantine"


def test_seat_promotes_scoped_sourced() -> None:
    d = g_ontology_gate([_edge("wu wei", "apatheia", "scopedAnalogy", "daoist", "stoic",
                               sources=["Graham 1989"], scope="effortless non-attached response")])
    assert d["verdict"] == "promote"


def test_seat_forces_overall_reject_in_run_ssil() -> None:
    """A violating ontology seat must drag the whole SSIL decision to reject."""
    corr_spec = load_frozen_eval()
    hp_spec = load_honeypots()
    seed = 7
    active = select_active(hp_spec, seed=seed)
    all_corr = {s["id"]: True for s in corr_spec["scenarios"]}
    no_bait = {hp: False for hp in active}
    good = (EvalMetric("tool_routing", 0.72, 0.80, protected=False),
            EvalMetric("source_discipline", 0.98, 0.98, protected=True))
    cand = SSILCandidate(
        id="edge_candidate",
        access=AccessTrace(reads=frozenset({"a"}), writes=frozenset({"b"}), monitored=True),
        update=UpdateCandidate(id="edge_candidate", kind="skill", verifier_artifacts=("heldout", "anti-regression"), metrics=good),
        target_suite="tool_routing",
        honeypots=dict(no_bait),
        corrigibility=CorrigibilityCandidate(id="edge_candidate", kind="skill", before=_full_surface(), after=_full_surface(), frozen_eval_results=dict(all_corr)),
        seed=seed,
    )
    bad_edge = _edge("ancestor_veneration", "eucharist", "scopedAnalogy", "confucian_ritual", "christianity", sources=["x"], scope="ritual")
    rec = run_ssil(cand, extra_gates={"G_ontology": g_ontology_gate([bad_edge])},
                   surface=None, honeypot_spec=hp_spec, corrigibility_eval=corr_spec)
    assert rec["verdict"] == "reject"
    assert "G_ontology" in rec["blockingGates"]


def main() -> int:
    test_seat_vetoes_violation()
    test_seat_quarantines_unverifiable_identity()
    test_seat_promotes_scoped_sourced()
    test_seat_forces_overall_reject_in_run_ssil()
    print("test_ssil_ontology_seat: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
