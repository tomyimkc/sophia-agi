#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the per-domain empirical competence model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.competence_model import (
    build_competence_model,
    competence_gap_worklist,
    learning_priorities,
    reliability_diagram,
)


def _rec(domain, confidence, correct):
    return {"domain": domain, "confidence": confidence, "correct": correct}


# A well-calibrated domain: confidence tracks accuracy. High-confidence answers are
# right, low-confidence ones are wrong.
def _calibrated_records(domain, n_high=8, n_low=4):
    recs = []
    # 8 records at 0.9 confidence, all correct (matches ~0.9 accuracy in that bin).
    for i in range(n_high):
        recs.append(_rec(domain, 0.9, i != 0))  # 7/8 correct ~ 0.875
    # 4 records at 0.2 confidence, mostly wrong.
    for i in range(n_low):
        recs.append(_rec(domain, 0.2, i == 0))  # 1/4 correct = 0.25
    return recs


# A miscalibrated/overconfident domain: high confidence but frequently wrong.
def _overconfident_records(domain, n=12):
    recs = []
    for i in range(n):
        recs.append(_rec(domain, 0.9, i % 2 == 0))  # 0.9 stated, 0.5 actual
    return recs


def test_reliability_diagram_bins_sum_and_values():
    records = [
        _rec("philosophy", 0.05, False),
        _rec("philosophy", 0.15, True),
        _rec("philosophy", 0.95, True),
        _rec("philosophy", 0.92, True),
        _rec("philosophy", 0.91, False),
    ]
    diag = reliability_diagram(records, n_bins=10)
    # bin counts must sum to n
    assert sum(b["count"] for b in diag) == len(records)
    # bins emitted in ascending confidence order
    los = [b["binLo"] for b in diag]
    assert los == sorted(los)
    # the 0.0-0.1 bin: one record, confidence 0.05, accuracy 0 (False)
    b0 = next(b for b in diag if b["binLo"] == 0.0)
    assert b0["count"] == 1
    assert b0["meanConfidence"] == 0.05
    assert b0["accuracy"] == 0.0
    # the 0.9-1.0 bin: three records (0.95,0.92,0.91), 2 correct => accuracy 2/3
    b9 = next(b for b in diag if b["binLo"] == 0.9)
    assert b9["count"] == 3
    assert b9["accuracy"] == round(2 / 3, 4)
    assert b9["meanConfidence"] == round((0.95 + 0.92 + 0.91) / 3, 4)


def test_calibrated_beats_overconfident():
    records = _calibrated_records("philosophy") + _overconfident_records("psychology")
    model = build_competence_model(records, alpha=0.1, coverage=0.5)
    cal = model.competence("philosophy")
    over = model.competence("psychology")
    assert cal > over, (cal, over)
    # The miscalibrated domain must show a non-trivial ECE.
    assert model.get("psychology").ece > model.get("philosophy").ece


def test_learning_priorities_ranks_weak_above_strong():
    # strong domain: high accuracy + calibrated; weak domain: overconfident + wrong.
    strong = _calibrated_records("history")
    weak = _overconfident_records("religion")
    model = build_competence_model(strong + weak, alpha=0.1, coverage=0.5)
    pr = learning_priorities(model)
    order = [p["domain"] for p in pr]
    assert order.index("religion") < order.index("history"), order
    # the weak domain carries a higher deficit
    deficits = {p["domain"]: p["deficit"] for p in pr}
    assert deficits["religion"] > deficits["history"]
    assert "domain" in pr[0] and "competence" in pr[0] and "reasons" in pr[0]


def test_selective_risk_below_base_when_confidence_informative():
    # Confidence is informative: high-conf correct, low-conf wrong => selective < base.
    records = []
    for _ in range(6):
        records.append(_rec("history", 0.95, True))
    for _ in range(4):
        records.append(_rec("history", 0.10, False))
    model = build_competence_model(records, alpha=0.1, coverage=0.5)
    dc = model.get("history")
    assert dc.selectiveRisk < dc.baseRisk, (dc.selectiveRisk, dc.baseRisk)
    assert dc.selectiveBeatsBase is True


def test_unseen_domain_fails_closed():
    model = build_competence_model(_calibrated_records("philosophy"), alpha=0.1, coverage=0.5)
    # never-seen domain: lowest competence, most conservative threshold
    assert model.competence("history") == 0.0
    assert model.threshold("history") == 0.0
    dc = model.get("history")
    assert dc.seen is False
    assert dc.baseRisk == 1.0
    # and it is treated as maximally weak in the unseen lookup record
    assert dc.competence == 0.0


def test_nonconformity_derived_when_absent():
    # No explicit nonconformity => derived as 1 - confidence; conformal fit succeeds.
    model = build_competence_model(_calibrated_records("philosophy"), alpha=0.1, coverage=0.5)
    dc = model.get("philosophy")
    assert 0.0 <= dc.threshold <= 1.0
    assert dc.nCalibration > 0


def test_to_dict_schema_and_candidate_only():
    model = build_competence_model(_calibrated_records("philosophy"))
    d = model.to_dict()
    assert d["schema"] == "sophia.competence_model.v1"
    assert d["candidateOnly"] is True
    assert "philosophy" in d["domains"]


def test_competence_gap_worklist_bridge():
    model = build_competence_model(
        _calibrated_records("history") + _overconfident_records("religion")
    )
    wl = competence_gap_worklist(model, thin_targets=("ritual",))
    assert wl["schema"] == "sophia.competence_gap_worklist.v1"
    assert wl["candidateOnly"] is True
    domains = [p["domain"] for p in wl["competencePriorities"]]
    assert domains.index("religion") < domains.index("history")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")
