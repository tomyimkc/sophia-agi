#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the declared judge-consensus policy (the ≥2-family gate as an object)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.consensus import (  # noqa: E402
    VALIDATION_POLICY,
    JudgeConsensusPolicy,
)


def _full_pass_kwargs() -> dict:
    return dict(families=["qwen", "mlx"], seeds=[0, 1, 2], agreement=0.55,
                judge_lineages=["qwen", "llama"], subject_lineage="olmoe")


def test_validation_policy_matches_the_published_gate() -> None:
    p = VALIDATION_POLICY
    assert (p.min_families, p.min_seeds, p.min_agreement) == (2, 3, 0.40)
    assert p.agreement_metric == "cohen_kappa"
    assert p.validate() == []
    assert p.evaluate(**_full_pass_kwargs())["ok"]


def test_each_clause_fails_specifically() -> None:
    p = VALIDATION_POLICY
    base = _full_pass_kwargs()
    cases = {
        "families": {**base, "families": ["qwen", "qwen"]},
        "seeds": {**base, "seeds": [0, 0, 1]},
        "agreement": {**base, "agreement": 0.2},
        "judge lineage equals subject": {**base, "judge_lineages": ["olmoe", "mlx"]},
    }
    for needle, kwargs in cases.items():
        verdict = p.evaluate(**kwargs)
        assert not verdict["ok"], needle
        assert any(needle.split()[0] in f for f in verdict["failures"]), verdict


def test_missing_inputs_abstain_fail_closed() -> None:
    p = VALIDATION_POLICY
    v = p.evaluate(families=["qwen", "mlx"], seeds=[0, 1, 2], agreement=None,
                   judge_lineages=None, subject_lineage=None)
    assert not v["ok"]
    joined = " ".join(v["failures"])
    assert "not reported" in joined and "lineages not declared" in joined


def test_metric_substitution_must_be_labelled() -> None:
    v = VALIDATION_POLICY.evaluate(**{**_full_pass_kwargs(),
                                      "agreement_metric": "gwet_ac1"})
    assert not v["ok"] and any("fallback" in f for f in v["failures"])


def test_weakened_policy_is_self_invalidating() -> None:
    weak = JudgeConsensusPolicy(min_families=1, min_seeds=1, tie_rule="majority")
    problems = weak.validate()
    assert len(problems) >= 3
    # and evaluate carries those problems into every verdict (can't quietly use it)
    assert not weak.evaluate(**_full_pass_kwargs())["ok"]
