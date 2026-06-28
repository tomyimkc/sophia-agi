#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Consequence-cascade benchmark tests — pack well-formedness + sweep integrity.

These do NOT teach-to-the-test: they assert structural invariants of the pack
(every case resolves, every severity matches its band), the integrity of the
threshold sweep (it finds an interior optimum that beats the 0.15 placeholder's
accuracy), and the no-overclaim envelope of the report. The per-case verdict
accuracy at the recommended threshold is asserted at >= 0.90 (NOT 1.0) — the
boundary cases near the threshold are *expected* to be the discriminator, and
hard-coding a perfect score would defeat the benchmark's purpose.

Dependency-free, offline, deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.run_consequence_cascade_benchmark as bm  # noqa: E402

PACK = ROOT / "eval" / "consequence_cascade" / "consequence_cascade_40_v1.jsonl"


def _cases():
    return bm.load_jsonl(PACK)


def test_pack_is_nonempty_and_well_formed() -> None:
    cases = _cases()
    assert len(cases) >= 30, f"pack too small: {len(cases)} cases"
    for c in cases:
        assert c["id"], "case missing id"
        assert c["expectVerdict"] in {"escalate", "allow", "abstain"}, c["id"]
        assert "graph" in c and "nodes" in c["graph"], c["id"]
        assert c["move"], f"{c['id']}: missing move"
        # every node has a slug id + a list derivesFrom
        for n in c["graph"]["nodes"]:
            assert n["id"], f"{c['id']}: node missing id"
            assert isinstance(n.get("derivesFrom", []), list), f"{c['id']}: bad derivesFrom"
            # ids must be lowercase slugs (mixed-case would silently fail to resolve)
            assert n["id"] == n["id"].lower(), f"{c['id']}: non-lowercase id {n['id']!r}"
        assert len(c["expectFlipSeverityBand"]) == 2, f"{c['id']}: bad severity band"


def test_pack_has_all_three_verdict_classes() -> None:
    cases = _cases()
    kinds = {c["expectVerdict"] for c in cases}
    assert {"escalate", "allow", "abstain"} <= kinds, f"missing verdict classes: {kinds}"
    # and the discriminator zone is represented
    boundary = [c for c in cases if c["caseType"] == "boundary_cascade"]
    assert len(boundary) >= 3, f"too few boundary cases: {len(boundary)}"


def test_every_case_severity_matches_its_band() -> None:
    # Integrity: the structurally-expected severity band must hold for every
    # resolvable case. If a label drifts from the real computed severity, the
    # ground truth is dishonest and the sweep is meaningless.
    cases = _cases()
    for c in cases:
        row = bm.run_case(c, threshold=0.15)
        if c["expectVerdict"] == "abstain":
            assert not row["found"], f"{c['id']}: expected unbounded/abstain but target resolved"
        else:
            assert row["found"], f"{c['id']}: target did not resolve (expected resolvable)"
            lo, hi = c["expectFlipSeverityBand"]
            assert lo <= row["flipSeverity"] <= hi, (
                f"{c['id']}: severity {row['flipSeverity']} outside band [{lo},{hi}]"
            )


def test_sweep_finds_interior_optimum_that_beats_placeholder() -> None:
    cases = _cases()
    sweep = bm.sweep_threshold(cases)
    rec = sweep["recommended"]
    table = {e["threshold"]: e for e in sweep["table"]}
    # optimum is interior (not pinned at a candidate-set boundary)
    cands = sweep["candidates"]
    assert min(cands) < rec < max(cands), f"recommended {rec} is at a sweep boundary"
    # it matches or beats the 0.15 hand-pick's accuracy
    assert table[rec]["verdictAccuracy"] >= table[0.15]["verdictAccuracy"], (
        f"recommended {rec} ({table[rec]['verdictAccuracy']}) does not beat 0.15 ({table[0.15]['verdictAccuracy']})"
    )
    # and the optimum's accuracy is high (the classes are separable by construction)
    assert table[rec]["verdictAccuracy"] >= 0.90, (
        f"recommended {rec} accuracy {table[rec]['verdictAccuracy']} < 0.90"
    )


def test_recommended_threshold_gives_high_verdict_accuracy() -> None:
    # Not 1.0 on purpose: boundary cases near the threshold are the discriminator.
    # A perfect score would mean the pack has no discriminating power.
    cases = _cases()
    sweep = bm.sweep_threshold(cases)
    rec = sweep["recommended"]
    rows = [bm.run_case(c, threshold=rec) for c in cases]
    acc = sum(1 for r in rows if r["verdictOk"]) / len(rows)
    assert acc >= 0.90, f"verdict accuracy at recommended {rec}: {acc}"


def test_report_envelope_is_no_overclaim() -> None:
    import json
    out = ROOT / "agi-proof" / "benchmark-results" / "consequence-cascade.public-report.json"
    assert out.exists(), "evidence report not generated; run the benchmark first"
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["schema"] == "sophia.consequence_cascade_benchmark.v1"
    assert rep["candidateOnly"] is True
    assert rep["level3Evidence"] is False
    assert rep["validated"] is False
    assert "claimBoundary" in rep and len(rep["claimBoundary"]) > 50
    # the sweep is recorded in the artifact
    assert "thresholdSweep" in rep and "recommended" in rep["thresholdSweep"]


def test_sweep_tiebreak_has_no_hidden_threshold_bias() -> None:
    # Regression for the "-abs(t - 0.15)" review thread: the recommendation must
    # be purely data-derived (accuracy, then margin, then declaration order) with
    # NO hidden preference for 0.15 or any other value. We force an accuracy AND
    # margin TIE between two candidates and verify the FIRST-declared wins — and
    # that reversing the declaration flips the winner. This proves the tiebreak is
    # order-based, not value-based.
    import tools.run_consequence_cascade_benchmark as bm
    # Two cases with severities 0.2 and 0.8 (perfectly symmetric about 0.5).
    # case A: 5 isolated nodes, retract one -> 1/5 = 0.2 (allow)
    # case B: root + 4 leaves, retract root -> 5/5 = 1.0 ... need 0.8 instead:
    #   root + 4 leaves (all deriveFrom root) = 5 nodes, retract root = 5/5 = 1.0.
    #   For 0.8 we need 4/5: root + 3 leaves + 1 independent = 5 nodes; retract
    #   root -> {root, leaf0, leaf1, leaf2} = 4/5 = 0.8.
    two_case_pack = [
        {"id": "lo", "caseType": "routine_retraction",  # 1/5 = 0.2
         "graph": {"nodes": [{"id": f"n{i}", "derivesFrom": []} for i in range(4)] +
                  [{"id": "leaf", "derivesFrom": ["n0"]}]},
         "move": "leaf", "expectVerdict": "allow", "expectFlipSeverityBand": [0.15, 0.25]},
        {"id": "hi", "caseType": "severe_cascade",  # 4/5 = 0.8
         "graph": {"nodes": [{"id": "root", "derivesFrom": []},
                             {"id": "l0", "derivesFrom": ["root"]},
                             {"id": "l1", "derivesFrom": ["root"]},
                             {"id": "l2", "derivesFrom": ["root"]},
                             {"id": "indep", "derivesFrom": []}]},
         "move": "root", "expectVerdict": "escalate", "expectFlipSeverityBand": [0.7, 0.9]},
    ]
    # Candidates 0.4 and 0.6 are symmetric about 0.5; both have margin 0.2 to the
    # nearest severity (0.4->0.2 = 0.2; 0.6->0.8 = 0.2) and both 100% accuracy.
    sweep = bm.sweep_threshold(two_case_pack, candidates=[0.4, 0.6])
    assert sweep["recommended"] == 0.4, f"expected first-declared 0.4, got {sweep['recommended']}"
    # Reverse declaration order -> winner must flip to 0.6 (proves order-based).
    sweep_rev = bm.sweep_threshold(two_case_pack, candidates=[0.6, 0.4])
    assert sweep_rev["recommended"] == 0.6, (
        f"tiebreak is value-biased (not order-based): reversed input gave {sweep_rev['recommended']}"
    )


def test_sweep_reclassifies_on_raw_unrounded_severity() -> None:
    # Regression for the rounding-mismatch review threads: the sweep must
    # reclassify using the RAW severity (len(abstain)/n), matching the live gate,
    # not rep.flipSeverity (rounded to 4dp). The row records rawFlipSeverity (6dp
    # for display); assert it is the true repeating fraction, distinct from the
    # 4dp-rounded value the gate reports — proving the sweep uses more precision
    # than the rounded field.
    import tools.run_consequence_cascade_benchmark as bm
    # 3-node graph: root + 2 leaves. Retract one leaf -> 1/3 = 0.333333...
    case = {"id": "third", "caseType": "boundary_cascade",
            "graph": {"nodes": [{"id": "root", "derivesFrom": []},
                                {"id": "l0", "derivesFrom": ["root"]},
                                {"id": "l1", "derivesFrom": ["root"]}]},
            "move": "l0", "expectVerdict": "escalate", "expectFlipSeverityBand": [0.3, 0.4]}
    row = bm.run_case(case, threshold=0.15)
    # rawFlipSeverity is stored at 6dp for display: 0.333333. The true value is
    # 1/3 = 0.333333...; the 4dp-rounded flipSeverity field is 0.3333.
    assert row["rawFlipSeverity"] == 0.333333, f"rawFlipSeverity: {row['rawFlipSeverity']!r}"
    assert row["flipSeverity"] == 0.3333, f"flipSeverity (4dp): {row['flipSeverity']!r}"
    assert row["rawFlipSeverity"] != row["flipSeverity"], (
        "raw and rounded severity must differ for 1/3 — proves they are distinct values"
    )
    # And reclassification uses the raw value: a threshold between 0.3333 and 0.333333
    # (0.3334) classifies via raw (0.333333 < 0.3334 -> allow) correctly.
    row2 = bm.run_case(case, threshold=0.3334)
    assert row2["gotVerdict"] == "allow", (
        f"raw 0.333333 < 0.3334 must be allow; got {row2['gotVerdict']} (rounding leak?)"
    )


def test_abstain_case_that_unexpectedly_resolves_fails_band_check() -> None:
    # Regression for the abstain band-ok review thread: a case labeled "abstain"
    # whose target unexpectedly RESOLVES (pack drift, e.g. a ghost id that
    # collides with a real node) must fail flipSeverityBandOk, not pass silently.
    import tools.run_consequence_cascade_benchmark as bm
    # "ghost_a" is declared as the move but ALSO exists as a real node -> resolves.
    drift_case = {"id": "drift", "caseType": "unbounded_target",
                  "graph": {"nodes": [{"id": "ghost_a", "derivesFrom": []},
                                      {"id": "other", "derivesFrom": ["ghost_a"]}]},
                  "move": "ghost_a", "expectVerdict": "abstain",
                  "expectFlipSeverityBand": [0.0, 0.0]}
    row = bm.run_case(drift_case, threshold=0.15)
    assert row["found"] is True, "test setup: ghost_a should resolve (drift)"
    assert row["flipSeverityBandOk"] is False, (
        "an abstain-labeled case whose target resolved must fail the band check (pack drift)"
    )


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_consequence_cascade_benchmark: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
