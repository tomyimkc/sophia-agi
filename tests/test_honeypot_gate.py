#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.honeypot_gate import (  # noqa: E402
    HoneypotCandidate,
    demo_honeypot_report,
    evaluate_honeypots,
    load_honeypots,
    select_active,
)

SPEC = load_honeypots()


def test_active_set_covers_all_required_modes() -> None:
    for seed in range(6):
        active = select_active(SPEC, seed=seed)
        by_id = {h["id"]: h for h in SPEC["honeypots"]}
        modes = sorted({by_id[h]["goodhartMode"] for h in active})
        assert modes == sorted(SPEC["requireModes"]), (seed, modes)


def test_rotation_varies_with_seed() -> None:
    sets = {tuple(select_active(SPEC, seed=s)) for s in range(8)}
    # Rotation should produce more than one distinct active set across seeds.
    assert len(sets) > 1


def test_rotation_is_deterministic() -> None:
    assert select_active(SPEC, seed=3) == select_active(SPEC, seed=3)


def test_honest_candidate_promotes() -> None:
    active = select_active(SPEC, seed=1)
    c = HoneypotCandidate(id="h", results={hp: False for hp in active})
    assert evaluate_honeypots(c, spec=SPEC, seed=1).verdict == "promote"


def test_bait_taker_rejects() -> None:
    active = select_active(SPEC, seed=1)
    c = HoneypotCandidate(id="b", results={**{hp: False for hp in active}, active[0]: True})
    d = evaluate_honeypots(c, spec=SPEC, seed=1)
    assert d.verdict == "reject"
    assert any("bait" in r for r in d.reasons)


def test_partial_eval_quarantines() -> None:
    active = select_active(SPEC, seed=1)
    c = HoneypotCandidate(id="p", results={hp: False for hp in active[:-1]})
    d = evaluate_honeypots(c, spec=SPEC, seed=1)
    assert d.verdict == "quarantine"
    assert any("unanswered" in r for r in d.reasons)


def test_probing_non_active_rejects() -> None:
    active = select_active(SPEC, seed=1)
    all_ids = [h["id"] for h in SPEC["honeypots"]]
    non_active = [i for i in all_ids if i not in active]
    assert non_active, "expected some non-active honeypots in the pool"
    c = HoneypotCandidate(id="probe", results={**{hp: False for hp in active}, non_active[0]: False})
    d = evaluate_honeypots(c, spec=SPEC, seed=1)
    assert d.verdict == "reject"
    assert any("probing" in r for r in d.reasons)


def test_bait_dominates_partial() -> None:
    active = select_active(SPEC, seed=2)
    c = HoneypotCandidate(id="bp", results={active[0]: True})  # took bait, rest unanswered
    assert evaluate_honeypots(c, spec=SPEC, seed=2).verdict == "reject"


def test_demo_invariants() -> None:
    rep = demo_honeypot_report()
    assert all(rep["invariants"].values()), rep["invariants"]
    assert rep["candidateOnly"] is True
    assert rep["level3Evidence"] is False


def main() -> int:
    test_active_set_covers_all_required_modes()
    test_rotation_varies_with_seed()
    test_rotation_is_deterministic()
    test_honest_candidate_promotes()
    test_bait_taker_rejects()
    test_partial_eval_quarantines()
    test_probing_non_active_rejects()
    test_bait_dominates_partial()
    test_demo_invariants()
    print("test_honeypot_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
