#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""On-pod multi-seed execution modes for tools/runpod_train.py.

Pure script-generation + arg tests (no pod, no network): Mode A (parallel, one
seed per GPU), Mode B (sequential, one GPU), and single-seed backward compat.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runpod_train import (  # noqa: E402
    _effective_gpu_count,
    _remote_train_script,
    _seeds_list,
    parse_args,
)


def _args(extra: list[str]):
    return parse_args(["--dry-run", "--train-data", "pack.jsonl", "--train-only", *extra])


def test_seeds_list_single_vs_multi() -> None:
    assert _seeds_list(_args([])) == [0]
    assert _seeds_list(_args(["--seed", "2"])) == [2]
    assert _seeds_list(_args(["--seeds", "0,1,2"])) == [0, 1, 2]


def test_mode_a_parallel_one_gpu_per_seed() -> None:
    args = _args(["--seeds", "0,1,2", "--on-pod-mode", "parallel", "--adapter-dir", "/ws/cuda-v1"])
    assert _effective_gpu_count(args) == 3  # one GPU per seed
    script = _remote_train_script(args)
    for i in (0, 1, 2):
        assert f"CUDA_VISIBLE_DEVICES={i}" in script
        assert f"--seed {i}" in script
        assert f"/ws/cuda-v1-seed{i}" in script
        assert f"sophia-cuda-v1-seed{i}.tar.gz" in script
    assert script.count("&\n") >= 3 and "\nwait\n" in script  # backgrounded then waited
    # on-pod multi-seed returns adapters only — no eval/promote on the pod
    assert "promote_adapter.py" not in script
    assert "eval_ladder.py" not in script


def test_mode_b_sequential_single_gpu() -> None:
    args = _args(["--seeds", "0,1,2", "--on-pod-mode", "sequential"])
    assert _effective_gpu_count(args) == 1  # sequential shares one GPU
    script = _remote_train_script(args)
    assert "CUDA_VISIBLE_DEVICES=" not in script  # no per-GPU pinning
    assert "\nwait\n" not in script  # not backgrounded
    assert script.count("train_lora.py") == 3  # three seeds, back-to-back
    assert "| tee " in script


def test_single_seed_backward_compatible() -> None:
    args = parse_args(["--dry-run", "--train-data", "pack.jsonl", "--train-only", "--seed", "0"])
    assert _effective_gpu_count(args) == 1
    script = _remote_train_script(args)
    assert "Sophia real training run complete." in script  # single-seed path
    assert "multi-seed run complete" not in script


def main() -> int:
    test_seeds_list_single_vs_multi()
    test_mode_a_parallel_one_gpu_per_seed()
    test_mode_b_sequential_single_gpu()
    test_single_seed_backward_compatible()
    print("test_runpod_pod_modes: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
