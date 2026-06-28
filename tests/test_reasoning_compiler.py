#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the 'reasoning compiler' thesis (feature #3).

Pure stdlib, deterministic, offline. The central check is the compiler-correctness
property: optimization passes (CSE + dead-code elimination) cut verification cost but leave
the goal's grounded conclusion invariant; type-check catches contradictions fail-closed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.reasoning_compiler import (  # noqa: E402
    Claim,
    ReasoningGraph,
    compile_graph,
    effective_confidence,
    is_grounded,
    main,
    pass_canonicalize,
    pass_dead_code_elimination,
    run_experiment,
)


def _chain():
    claims = {
        "s0": Claim("s0", "fact A", "source", 3, []),
        "s1": Claim("s1", "fact B", "source", 2, []),
        "d0": Claim("d0", "step P", "derived", 3, ["s0", "s1"]),
        "d0dup": Claim("d0dup", "step P", "derived", 3, ["s0", "s1"]),
        "dead": Claim("dead", "irrelevant", "derived", 1, ["s0"]),
        "g": Claim("g", "conclusion", "goal", 3, ["d0", "d0dup"]),
    }
    return ReasoningGraph(claims, "g")


def test_min_over_chain_matches_okf_semantics():
    g = _chain()
    # weakest link: goal -> step P -> min(3, A=3, B=2) = 2
    assert effective_confidence(g, "g") == 2
    assert is_grounded(g, "g")


def test_cse_merges_duplicate_claims():
    g = _chain()
    opt = pass_canonicalize(g)
    stmts = sorted(opt.claims[c].statement.lower() for c in opt.claims)
    assert stmts.count("step p") == 1  # the duplicate is folded


def test_dce_drops_dead_claims():
    g = _chain()
    opt = pass_dead_code_elimination(pass_canonicalize(g))
    norms = {opt.claims[c].statement.lower() for c in opt.claims}
    assert "irrelevant" not in norms


def test_passes_preserve_semantics_and_cut_cost():
    res = compile_graph(_chain())
    assert res.semantics_preserved
    assert res.goal_confidence_before == res.goal_confidence_after == 2
    assert res.cost_after < res.cost_before
    assert res.emittable


def test_contradiction_fails_closed():
    claims = {
        "s0": Claim("s0", "fact A", "source", 3, []),
        "d0": Claim("d0", "step P", "derived", 3, ["s0"]),
        "neg": Claim("neg", "not step P", "derived", 3, ["s0"]),
        "g": Claim("g", "conclusion", "goal", 3, ["d0", "neg"]),
    }
    res = compile_graph(ReasoningGraph(claims, "g"))
    assert res.contradictions
    assert not res.emittable  # never emit on a contradicted premise


def test_experiment_confirms_all_hypotheses():
    r = run_experiment(graphs=200, seed=5)
    assert r["mean_cost_reduction"] > 0.0            # H1
    assert r["semantics_preserved_rate"] == 1.0      # H2
    assert r["dce_exact_rate"] == 1.0
    assert r["contradiction_recall"] == 1.0          # H3
    assert r["failclosed_rate"] == 1.0
    assert r["false_contradiction_rate"] == 0.0


def test_determinism():
    assert run_experiment(graphs=120, seed=9) == run_experiment(graphs=120, seed=9)


def test_cli():
    assert main(["--self-test"]) == 0
    assert main(["--run", "--graphs", "80"]) == 0
    assert main(["--json", "--graphs", "80"]) == 0
