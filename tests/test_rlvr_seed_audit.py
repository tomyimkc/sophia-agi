#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seed forensics for the RLVR sweep (measurement-artifact regression tests).

Context (six-run gate/multiaxis sweep on rlvr-runpod.yml): two runs with a
DIFFERENT reward arm AND a DIFFERENT seed came back with byte-identical
adapter-eval numbers, and the seed->base mapping did not reconcile across arms.
The dispatched seed WAS passed textually to both train and eval, but nothing
bound the copied-back report to the run that dispatched it: the pod wrote the
eval to a FIXED path on a reusable/shared /workspace, an eval failure was
swallowed non-fatally, and the scp/ingest chain had no identity check — so a
stale or concurrent run's report could be attributed to this run's (arm, seed).

These offline tests pin the fixes and FAIL under the old behavior:

  1. different seeds provably yield different held-out splits;
  2. every adapter-eval report records ``audit.effectiveSeed`` and a
     ``splitHash`` over the exact scored ids (identity evidence, candidateOnly);
  3. the real-mode eval path passes the dispatched seed into the generator
     loader (eval generation is seeded, not just the split);
  4. the pod script writes the eval to a (task,reward,seed)-stamped path and
     clears it BEFORE the eval, so a swallowed failure can never resurface a
     stale report (fail-closed: no fresh eval -> no file).

No capability claims are made or implied here; this file tests measurement
identity only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import rl_dataset  # noqa: E402
from tools import eval_rlvr_adapter as era  # noqa: E402
from tools import runpod_rlvr  # noqa: E402


def _eval_args(seed: int, **overrides) -> argparse.Namespace:
    ns = argparse.Namespace(
        mode="mock", task="provenance", model="mock", adapter=None,
        seed=seed, eval_frac=0.3, limit=0, max_new_tokens=8,
        max_fp_regression=0.0, capability_panel=False, step_domain="math",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# --------------------------------------------------------------------------- #
# 1. Different seeds -> different held-out splits (the seed is load-bearing).
# --------------------------------------------------------------------------- #
def test_different_seeds_yield_different_heldout_splits() -> None:
    sealed = {}
    ids = {}
    for seed in (0, 1, 2):
        data = rl_dataset.build_rl_dataset(eval_frac=0.3, seed=seed)
        sealed[seed] = data["eval_sealed"]
        ids[seed] = sorted(c.id for c in data["eval_cases"])
    assert len(set(sealed.values())) == 3, f"seeds 0/1/2 must give distinct eval splits: {sealed}"
    assert ids[0] != ids[1] and ids[1] != ids[2] and ids[0] != ids[2]
    # And the same seed is reproducible (identity, not noise).
    again = rl_dataset.build_rl_dataset(eval_frac=0.3, seed=1)
    assert again["eval_sealed"] == sealed[1]


# --------------------------------------------------------------------------- #
# 2. The adapter-eval report RECORDS its identity: effectiveSeed + splitHash.
#    Fails under the old behavior (no ``audit`` key in the report).
# --------------------------------------------------------------------------- #
def test_report_records_effective_seed_and_split_hash() -> None:
    r0 = era.run_eval(_eval_args(0))
    r1 = era.run_eval(_eval_args(1))
    for seed, rep in ((0, r0), (1, r1)):
        audit = rep.get("audit")
        assert audit, "adapter-eval report must carry an audit block (old behavior: absent)"
        assert audit["effectiveSeed"] == seed
        assert audit["evalFrac"] == 0.3
        assert audit["splitHash"] and audit["nEvalIds"] > 0
    # Different dispatched seeds -> different recorded split identity.
    assert r0["audit"]["splitHash"] != r1["audit"]["splitHash"], (
        "two seeds reporting the SAME splitHash is exactly the sweep anomaly "
        "(stale/cross-run report pickup or an unplumbed seed)"
    )
    # The hash is verifiable against the dataset builder (evidence, not decoration).
    data = rl_dataset.build_rl_dataset(eval_frac=0.3, seed=1)
    expect = hashlib.sha256(
        json.dumps(sorted(c.id for c in data["eval_cases"])).encode("utf-8")
    ).hexdigest()[:16]
    assert r1["audit"]["splitHash"] == expect


def test_audit_block_present_for_all_mock_tasks() -> None:
    """Every task lane records the same audit contract (step/invention previously
    had no sealed-split evidence at all)."""
    runners = {
        "math": era.run_eval_math,
        "step": era.run_eval_step,
        "code": era.run_eval_code,
        "concept": era.run_eval_concept,
    }
    for task, fn in runners.items():
        rep = fn(_eval_args(1))
        audit = rep.get("audit")
        assert audit and audit["effectiveSeed"] == 1, f"{task}: missing/incorrect audit block"
        assert audit["splitHash"], f"{task}: missing splitHash"


# --------------------------------------------------------------------------- #
# 3. Real-mode wiring: the dispatched seed reaches the generator loader (eval
#    GENERATION is seeded, not only the split). Fails under the old behavior
#    (loader had no seed parameter; call sites passed none).
# --------------------------------------------------------------------------- #
def test_real_eval_passes_dispatched_seed_to_generators() -> None:
    seen: dict = {}

    def fake_loader(model, adapter, *, max_new_tokens, chat_template=False, seed=None):
        seen["seed"] = seed

        def gen(prompt: str) -> str:
            return prompt  # echo; scoring is irrelevant to this wiring assertion

        return gen, gen

    orig = era._load_real_generators
    try:
        era._load_real_generators = fake_loader  # type: ignore[assignment]
        rep = era.run_eval(_eval_args(7, mode="real", adapter=Path("/nonexistent-adapter"), limit=2))
    finally:
        era._load_real_generators = orig  # type: ignore[assignment]
    assert seen.get("seed") == 7, "run_eval must pass args.seed into _load_real_generators"
    assert rep["audit"]["effectiveSeed"] == 7


# --------------------------------------------------------------------------- #
# 4. Pod script: run-identity-stamped eval output, cleared BEFORE the eval.
#    Fails under the old behavior (fixed sophia-rlvr-v1.adapter-eval.json path,
#    never cleared -> a stale report survives a swallowed eval failure and is
#    scp'd back as this run's numbers).
# --------------------------------------------------------------------------- #
def test_remote_script_stamps_and_clears_adapter_eval_out() -> None:
    args = runpod_rlvr.parse_args([
        "--dry-run", "--source", "git", "--remote-mode", "live",
        "--task", "provenance", "--reward", "multiaxis", "--seed", "2",
    ])
    script = runpod_rlvr._remote_training_script(args)
    stamp = 'ADAPTER_EVAL_OUT="/workspace/sophia-runpod/sophia-rlvr-v1.adapter-eval.$SOPHIA_TASK-$SOPHIA_REWARD-seed$SOPHIA_SEED.json"'
    assert stamp in script, "adapter-eval out path must embed task/reward/seed run identity"
    assert 'rm -f "$ADAPTER_EVAL_OUT"' in script, "stale report must be cleared before the eval runs"
    assert script.index('rm -f "$ADAPTER_EVAL_OUT"') < script.index("eval_rlvr_adapter.py"), (
        "the clear must happen BEFORE eval_rlvr_adapter is invoked (fail-closed)"
    )
    # The eval invocation still receives the dispatched seed and the stamped out path.
    assert '--seed "$SOPHIA_SEED"' in script
    assert '--out "$ADAPTER_EVAL_OUT"' in script
    # The old run-agnostic fixed path must be gone from the live script.
    assert "sophia-rlvr-v1.adapter-eval.json" not in script
    # Same guard on the code task's invention eval lane.
    assert 'INVENTION_EVAL_OUT="/workspace/sophia-runpod/sophia-rlvr-v1.invention-eval.seed$SOPHIA_SEED.json"' in script
    assert 'rm -f "$INVENTION_EVAL_OUT"' in script


def main() -> int:
    test_different_seeds_yield_different_heldout_splits()
    test_report_records_effective_seed_and_split_hash()
    test_audit_block_present_for_all_mock_tasks()
    test_real_eval_passes_dispatched_seed_to_generators()
    test_remote_script_stamps_and_clears_adapter_eval_out()
    print("test_rlvr_seed_audit: all tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
