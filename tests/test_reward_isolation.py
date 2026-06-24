#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reward_isolation import (  # noqa: E402
    AccessTrace,
    demo_reward_isolation_report,
    evaluate_reward_isolation,
    load_surface,
)

SPEC = load_surface()


def _ev(cid, trace):
    return evaluate_reward_isolation(cid, trace, surface=SPEC)


def test_clean_delta_promotes() -> None:
    t = AccessTrace(reads=frozenset({"agent/llm.py"}), writes=frozenset({"skills/x.py"}), monitored=True)
    assert _ev("clean", t).verdict == "promote"


def test_write_to_gate_rejects() -> None:
    t = AccessTrace(writes=frozenset({"agent/continual_plasticity.py"}), monitored=True)
    d = _ev("w", t)
    assert d.verdict == "reject"
    assert any("wrote protected" in r for r in d.reasons)


def test_write_to_constitution_rejects() -> None:
    t = AccessTrace(writes=frozenset({"constitution/constitution.v2.json"}), monitored=True)
    assert _ev("c", t).verdict == "reject"


def test_write_to_frozen_eval_rejects() -> None:
    t = AccessTrace(writes=frozenset({"eval/conscience/corrigibility_frozen.v1.json"}), monitored=True)
    assert _ev("f", t).verdict == "reject"


def test_write_to_ci_rejects() -> None:
    t = AccessTrace(writes=frozenset({".github/workflows/validate.yml"}), monitored=True)
    assert _ev("ci", t).verdict == "reject"


def test_reward_glob_write_rejects() -> None:
    t = AccessTrace(writes=frozenset({"provenance_bench/rl_reward.py"}), monitored=True)
    assert _ev("r", t).verdict == "reject"


def test_hidden_eval_read_rejects() -> None:
    t = AccessTrace(reads=frozenset({"private/hidden-evals/PACK.json"}), monitored=True)
    d = _ev("hr", t)
    assert d.verdict == "reject"
    assert any("blind" in r for r in d.reasons)


def test_tamper_pattern_rejects() -> None:
    t = AccessTrace(writes=frozenset({"skills/x.py"}), monitored=True, proposed_text="we should lower the threshold here")
    d = _ev("tp", t)
    assert d.verdict == "reject"
    assert any("tampering intent" in r for r in d.reasons)


def test_deep_path_globs_match() -> None:
    # ** patterns must catch protected paths at 2+ directory levels (correct glob semantics).
    from agent.reward_isolation import _matches
    surface = SPEC
    assert _matches(["a/b/c/secret_reward_config.py"], surface["noWriteGlobs"])      # **/*reward*
    assert _matches(["x/y/hidden-evals/pack.json"], surface["noReadGlobs"])           # **/hidden-evals/**
    assert _matches([".github/workflows/sub/deep.yml"], surface["noWriteGlobs"])      # .github/workflows/**
    assert _matches(["constitution/nested/dir/file.json"], surface["noWriteGlobs"])   # constitution/**
    # A clearly-unprotected deep path must NOT match.
    assert not _matches(["skills/sub/router.py"], surface["noWriteGlobs"])


def test_unmonitored_quarantines() -> None:
    t = AccessTrace(monitored=False)
    d = _ev("u", t)
    assert d.verdict == "quarantine"
    assert any("not monitored" in r for r in d.reasons)


def test_breach_beats_unmonitored() -> None:
    # A breach on an unmonitored trace still rejects (breach dominates).
    t = AccessTrace(writes=frozenset({"constitution/constitution.v2.json"}), monitored=False)
    assert _ev("b", t).verdict == "reject"


def test_demo_invariants() -> None:
    rep = demo_reward_isolation_report()
    assert all(rep["invariants"].values()), rep["invariants"]
    assert rep["candidateOnly"] is True
    assert rep["level3Evidence"] is False


def main() -> int:
    test_clean_delta_promotes()
    test_write_to_gate_rejects()
    test_write_to_constitution_rejects()
    test_write_to_frozen_eval_rejects()
    test_write_to_ci_rejects()
    test_reward_glob_write_rejects()
    test_hidden_eval_read_rejects()
    test_deep_path_globs_match()
    test_tamper_pattern_rejects()
    test_unmonitored_quarantines()
    test_breach_beats_unmonitored()
    test_demo_invariants()
    print("test_reward_isolation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
