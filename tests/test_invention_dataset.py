# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Open-invention instrument: it must measure composition the model hasn't seen.

The novelty pillar's validity claim is that eval pass-rate measures INVENTION
(composing seen primitives in an unseen order), not recall. These tests lock the
structural guarantees (disjoint compositions, primitive coverage, determinism) and
the discrimination proof (a memorizer fails eval while a deriver passes), and the
end-to-end composition with the hardened grader.
"""

from __future__ import annotations

import os

import pytest

from provenance_bench import invention_dataset as inv

EXEC_ON = os.environ.get("SOPHIA_ALLOW_CODE_EXEC", "0").strip().lower() in ("1", "true", "yes", "on")
exec_only = pytest.mark.skipif(not EXEC_ON, reason="needs SOPHIA_ALLOW_CODE_EXEC=1")


@pytest.mark.parametrize("depth", [2, 3])
def test_compositions_disjoint_and_primitives_covered(depth):
    data = inv.build_invention_dataset(depth=depth, seed=0)
    train = {tuple(c) for c in data["train_compositions"]}
    evalc = {tuple(c) for c in data["eval_compositions"]}
    assert train.isdisjoint(evalc)                      # eval combos never in train
    eval_prims = {p for c in evalc for p in c}
    train_prims = {p for c in train for p in c}
    assert eval_prims <= train_prims                    # but every piece IS seen
    assert evalc and train


def test_build_is_deterministic():
    a = inv.build_invention_dataset(depth=2, seed=1)
    b = inv.build_invention_dataset(depth=2, seed=1)
    assert a["eval_compositions"] == b["eval_compositions"]
    assert a["train_compositions"] == b["train_compositions"]


def test_seed_changes_split():
    a = inv.build_invention_dataset(depth=2, seed=0)
    b = inv.build_invention_dataset(depth=2, seed=2)
    assert a["eval_compositions"] != b["eval_compositions"]


@pytest.mark.parametrize("depth", [2, 3])
def test_discrimination_separates_invention_from_recall(depth):
    disc = inv.discrimination(depth=depth, seed=0)
    # The deriver composes novel pipelines; the memorizer cannot, yet it DOES know
    # the train compositions -> the eval gap is novelty, not incompetence.
    assert disc["deriver"]["derivation"] >= 0.99
    assert disc["memorizer"]["recall"] >= 0.95
    assert disc["memorizer"]["derivation"] <= 0.05
    assert disc["discriminates"] is True


@pytest.mark.parametrize("depth", [2, 3])
def test_offline_invariants_pass(depth):
    ok, detail = inv.offline_invariants(depth=depth)
    assert ok, detail


def test_depth_one_is_pure_recall_no_invention_split():
    # A single primitive cannot be both held-out AND seen-in-train, so depth 1 has
    # no novel-combination split — coverage repair empties eval. This is the honest
    # boundary: invention is only measurable at depth >= 2.
    data = inv.build_invention_dataset(depth=1, seed=0)
    assert data["eval_tasks"] == []
    ok, _ = inv.offline_invariants(depth=1)
    assert ok is False  # not a valid invention instrument at depth 1


def test_reference_solution_is_not_leaked_into_prompt_or_test():
    data = inv.build_invention_dataset(depth=2, seed=0)
    for t in data["eval_tasks"]:
        assert "reference_solution" in t
        assert "def pipeline" not in t["prompt"]         # spec, not solution
        assert "def pipeline" not in t["test"]           # asserts only
        assert t["test"].strip() and t["private_test"].strip()


def test_tool_check_returns_go():
    import tools.gen_invention_pack as g

    assert g.main(["--check", "--depth", "2"]) == 0


# --- composition with the hardened anti-cheat grader (CIC + novelty) --------

@exec_only
def test_reference_solution_passes_hardened_grader_on_holdout():
    from provenance_bench import code_integrity as ci

    data = inv.build_invention_dataset(depth=2, seed=0)
    for t in data["eval_tasks"][:5]:
        score, _ = ci.guarded_reward_for_task(
            t["reference_solution"], t["test"], holdout_test=t["private_test"])
        assert score == 1.0, t["id"]


@exec_only
def test_special_casing_cheat_fails_on_invention_task():
    from provenance_bench import code_integrity as ci

    data = inv.build_invention_dataset(depth=2, seed=0)
    t = data["eval_tasks"][0]
    ex = t["examples"][0]
    cheat = (f"```python\ndef pipeline(xs):\n    return {ex['output']!r} "
             f"if xs == {ex['input']!r} else []\n```")
    score, _ = ci.guarded_reward_for_task(cheat, t["test"], holdout_test=t["private_test"])
    assert score == -1.0
