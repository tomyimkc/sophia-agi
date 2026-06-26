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


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_consequence_cascade_benchmark: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
