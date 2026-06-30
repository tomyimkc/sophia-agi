#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for tools/judge_pool.py — the per-family multi-lane routing layer.

Pins: least-loaded routing is deterministic + balanced; validate_pool raises on a mixed-family
pool and on <2 families; families() matches the gate's _family_key; round-robin spreads load
across replicas. Offline, deterministic — no network, no GPU, no random.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.judge_pool import (  # noqa: E402
    load_pool, families, endpoints_for, next_endpoint, validate_pool,
    total_lanes, lanes_for_family, offline_invariants,
)
from tools.run_lora_uplift_validation import _family_key  # noqa: E402

# A 2-family pool: qwen (1 lane) + the 70B served as vllm vendor 'mlx-community' (3 lanes).
_POOL = {
    "qwen": ["vllm:Qwen/Qwen2.5-7B-Instruct@http://h0:8000/v1"],
    "mlx-community": [
        "vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@http://h1:8081/v1",
        "vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@http://h2:8001/v1",
        "vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@http://h3:8001/v1",
    ],
}


def test_self_test_passes() -> None:
    ok, detail = offline_invariants()
    assert ok, detail


def test_families_matches_family_key() -> None:
    """families() must be the DISTINCT _family_key set (the gate's keying), sorted."""
    fams = families(_POOL)
    assert fams == ["mlx-community", "qwen"]
    # every lane's family is one of the reported families, keyed identically to the gate
    for specs in _POOL.values():
        for spec in specs:
            assert _family_key(spec) in fams


def test_lane_counts() -> None:
    assert total_lanes(_POOL) == 4
    assert lanes_for_family(_POOL, "mlx-community") == 3
    assert lanes_for_family(_POOL, "qwen") == 1
    assert endpoints_for(_POOL, "nope") == []


def test_least_loaded_is_deterministic_and_tie_breaks_by_spec() -> None:
    lanes = _POOL["mlx-community"]
    load0 = {s: 0 for s in lanes}
    # all equal -> lexicographically smallest spec, deterministic across calls
    a = next_endpoint(_POOL, "mlx-community", load0)
    b = next_endpoint(_POOL, "mlx-community", load0)
    assert a == b == min(lanes)
    # the busiest lane is never chosen while a less-loaded one exists
    busy = {lanes[0]: 9, lanes[1]: 0, lanes[2]: 0}
    pick = next_endpoint(_POOL, "mlx-community", busy)
    assert pick != lanes[0]
    assert pick == min(lanes[1], lanes[2])


def test_round_robin_balances_load() -> None:
    """Repeatedly pick + increment -> load spreads evenly across the 3 lanes (no random)."""
    lanes = _POOL["mlx-community"]
    load = {s: 0 for s in lanes}
    picks = []
    for _ in range(12):  # 12 over 3 lanes -> 4 each
        p = next_endpoint(_POOL, "mlx-community", load)
        load[p] += 1
        picks.append(p)
    assert sorted(load.values()) == [4, 4, 4]
    assert set(picks) == set(lanes)  # every lane used


def test_validate_accepts_two_family_pool() -> None:
    ok, info = validate_pool(_POOL)
    assert ok
    assert info["families"] == ["mlx-community", "qwen"]
    assert info["totalLanes"] == 4
    assert info["lanesPerFamily"] == {"mlx-community": 3, "qwen": 1}


def test_validate_raises_on_under_two_families() -> None:
    one = {"mlx-community": _POOL["mlx-community"]}  # one family, 3 lanes
    raised = False
    try:
        validate_pool(one)
    except ValueError as e:
        raised = True
        assert "2" in str(e)  # mentions the 2-family requirement
    assert raised, "a 1-family pool (lanes != families) must be refused"


def test_validate_raises_on_mixed_family_label() -> None:
    """A config label whose replicas key to DIFFERENT families is a silent family-count corruption."""
    mixed = {
        "mlx-community": _POOL["mlx-community"] + ["vllm:Qwen/Qwen2.5-7B-Instruct@http://x:1/v1"],
        "qwen": _POOL["qwen"],
    }
    raised = False
    try:
        validate_pool(mixed)
    except ValueError as e:
        raised = True
        assert "mix" in str(e).lower() or "same" in str(e).lower()
    assert raised, "a label mixing families must be refused (adding LANES must not add FAMILIES)"


def test_load_pool_normalizes_shapes_and_validates_example_config() -> None:
    # both flat-list and {"replicas": [...]} shapes normalize to the same pool
    flat = load_pool({"families": {"a": ["mlx:m@http://h1/v1"], "b": ["ollama:q@http://h2/v1"]}})
    nested = load_pool({"families": {"a": {"replicas": ["mlx:m@http://h1/v1"]},
                                     "b": {"replicas": ["ollama:q@http://h2/v1"]}}})
    assert flat == nested
    # the shipped worked-example config loads + validates to 2 families / 4 lanes
    cfg = json.loads((ROOT / "config" / "inference.local.judge-pool.json").read_text("utf-8"))
    pool = load_pool(cfg)
    ok, info = validate_pool(pool)
    assert ok and info["totalLanes"] == 4 and info["families"] == ["mlx-community", "qwen"]


def test_load_pool_rejects_empty_and_malformed() -> None:
    for bad in ({}, {"families": {}}, {"families": {"a": []}}, {"families": {"a": "notalist"}}):
        raised = False
        try:
            load_pool(bad)
        except ValueError:
            raised = True
        assert raised, f"load_pool should reject {bad!r}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} judge_pool tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
