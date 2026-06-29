#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia-gated AutoResearch controller: firewall, power-gate, leakage, protected regression.

Deterministic, offline, no GPU — the controller is the brakes/odometer; the GPU training step
plugs in behind it as the experiment stream.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sophia_autoresearch import (  # noqa: E402
    Experiment,
    Measurement,
    decide,
    firewall_violations,
    offline_invariants,
    run_loop,
)


def _m(deltas, lower=True, metric="val_bpb"):
    return Measurement(metric, tuple(deltas), lower_is_better=lower)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_genuine_powered_win_is_kept() -> None:
    exp = Experiment("win", ("train.py",), _m([-0.05] * 12))
    d = decide(exp)
    assert d.verdict == "keep" and d.improved and d.ledger_entry is None


def test_greedy_point_estimate_without_power_is_discarded() -> None:
    # Mean is negative (looks like a win) but the CI straddles zero -> not kept (anti-overfit).
    exp = Experiment("noisy", ("train.py",), _m([-0.5, 0.4, -0.3, 0.45, -0.4, 0.5]))
    d = decide(exp)
    assert d.verdict == "discard"
    assert any("CI" in r for r in d.reasons)


def test_reward_hacking_firewall_rejects_verifier_edit() -> None:
    # Editing the thing that scores you is fatal, even with a spectacular metric.
    exp = Experiment("cheat", ("agent/gate.py",), _m([-0.9] * 12))
    d = decide(exp)
    assert d.verdict == "reject_tamper" and not d.kept
    assert d.ledger_entry is not None


def test_firewall_pattern_coverage() -> None:
    bad = firewall_violations([
        "agent/math_verifier.py", "provenance_bench/swarm_rl.py",
        "constitution/constitution.v2.json", "eval/code_provenance/holdout.jsonl",
        "data/religion_concepts.json", "train.py", "training/tool_use/dpo_pairs.jsonl",
    ])
    assert "train.py" not in bad
    assert "training/tool_use/dpo_pairs.jsonl" not in bad  # data IS editable
    assert "agent/math_verifier.py" in bad
    assert "constitution/constitution.v2.json" in bad
    assert "data/religion_concepts.json" in bad


def test_eval_leakage_is_discarded() -> None:
    exp = Experiment("leak", ("train.py",), _m([-0.2] * 12), decontaminated=False)
    d = decide(exp)
    assert d.verdict == "discard"
    assert any("decontam" in r for r in d.reasons)


def test_protected_regression_blocks_keep() -> None:
    exp = Experiment("reg", ("train.py",), _m([-0.2] * 12),
                     protected_regressions=("religion-attribution",))
    assert decide(exp).verdict == "discard"


def test_higher_is_better_metric_path() -> None:
    exp = Experiment("halluc", ("agent/swarm_router.py",),
                     _m([0.08] * 12, lower=False, metric="verified_halluc_delta"))
    assert decide(exp).verdict == "keep"


def test_every_non_keep_logs_a_ledger_entry() -> None:
    exps = [
        Experiment("win", ("train.py",), _m([-0.05] * 12)),
        Experiment("noisy", ("train.py",), _m([-0.5, 0.4, -0.3, 0.45])),
        Experiment("cheat", ("agent/gate.py",), _m([-0.9] * 12)),
    ]
    summary = run_loop(exps)
    assert summary["evaluated"] == 3
    assert summary["kept"] == 1
    # one ledger entry per non-keep
    assert len(summary["ledger"]) == 2


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} sophia_autoresearch tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
