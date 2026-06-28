#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the search-recall eval that unstubs the dual-use gate (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import search_recall as srk  # noqa: E402
from agent.dual_use_adapter import DualUseAdapter  # noqa: E402


def test_offline_invariants() -> None:
    ok, detail = srk.offline_invariants()
    assert ok, detail["checks"]


def test_recall_at_k_basic() -> None:
    assert srk.recall_at_k(["a", "b", "c"], ("a", "c"), 3) == 1.0
    assert srk.recall_at_k(["x", "b", "c"], ("a", "c"), 3) == 0.5
    assert srk.recall_at_k(["a"], (), 3) == 1.0  # empty gold = vacuous


def test_source_discipline_scorer() -> None:
    assert srk.source_discipline_ok("Traditionally attributed; authorship disputed (see source).")
    assert not srk.source_discipline_ok("He definitely wrote it.")
    assert not srk.source_discipline_ok("It was Laozi.")  # bare assertion, no grounding


def test_graded_gate_promotes_measured_gain() -> None:
    a = DualUseAdapter(id="theta-search-v1", team_name="search", gain=0.5)
    rep = srk.compare_discipline(
        srk.PACK_V1,
        before=lambda q: "He definitely wrote it.",
        after=lambda q: "Traditionally attributed but disputed; see the source.",
    )
    decision = srk.gate_from_scores(a, rep, verifier_artifacts=("recall_eval.json", "decontam.json"))
    assert decision.verdict == "promote"
    assert rep.delta > 0 and rep.is_win


def test_load_harder_pack_v2() -> None:
    pack = srk.load_pack()
    assert len(pack) >= 25  # the harder pack is larger than inline PACK_V1
    traps = [t for t in pack if t.trap]
    assert len(traps) >= 20  # more traps → more statistical power for the graded delta
    for t in pack:
        assert t.query and isinstance(t.gold_sources, tuple)
    # manifest agrees with the file
    import json
    man = json.loads((ROOT / "data" / "search_recall" / "manifest.json").read_text())
    assert man["n"] == len(pack) and man["nTraps"] == len(traps)


def test_pack_v2_drives_graded_gate() -> None:
    # On the bigger pack, a base that fabricates vs an adapter that hedges → measured win.
    pack = srk.load_pack()
    rep = srk.compare_discipline(
        pack,
        before=lambda q: "He definitely wrote it, without doubt.",
        after=lambda q: "Traditionally attributed but disputed; no reliable source — see the citation.",
    )
    assert rep.delta > 0 and rep.is_win


def test_no_false_win_identical_arms() -> None:
    gold = {t.query: t.gold_sources for t in srk.PACK_V1}
    strong = lambda q: list(gold.get(q, ())) + ["noise"]
    rep = srk.compare_recall(srk.PACK_V1, strong, strong, k=3)
    assert not rep.is_win and rep.delta == 0.0


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [ok] {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {fn.__name__}")
            traceback.print_exc()
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
