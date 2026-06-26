# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the local distillation driver + DPO-from-misses builder (Spark-5).
No GPU, no model load; exercises the plan builder + the DPO mining logic."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_distill_plan_stages() -> None:
    from tools import run_distill_local as drv

    args = types.SimpleNamespace(
        teacher="vllm:Qwen/Qwen2.5-14B-Instruct@http://localhost:8000/v1",
        student_model="Qwen/Qwen2.5-3B-Instruct", tasks=str(drv.DEFAULT_TASKS),
        traces_out=str(drv.DEFAULT_TRACES), adapter_out=str(drv.DEFAULT_ADAPTER),
        epochs=1, seed=0, limit=0, four_bit=False, dpo=False,
    )
    plan = drv.build_plan(args)
    stages = [name for name, _ in plan]
    assert stages == ["distill", "train"], stages
    assert plan[0][1][1].endswith("distill_council_traces.py")
    assert "--teacher" in plan[0][1]
    assert plan[1][1][1].endswith("train_lora.py") and "--rslora" in plan[1][1]

    args.dpo = True
    plan2 = drv.build_plan(args)
    assert [n for n, _ in plan2] == ["distill", "train", "dpo-pairs"]


def test_adapter_card_is_candidate_only() -> None:
    from tools import run_distill_local as drv

    args = types.SimpleNamespace(student_model="Qwen/Qwen2.5-3B-Instruct",
                                 teacher="vllm:Qwen/Qwen2.5-14B-Instruct@http://localhost:8000/v1",
                                 epochs=1, seed=0)
    card = drv._adapter_card(args, traces_count=42, adapter_dir=Path("x/local-sophia-distilled"))
    assert card["candidateOnly"] is True and card["canClaimAGI"] is False
    assert "DGX Spark" in card["claimBoundary"] and "x86" in card["claimBoundary"]
    assert card["training"]["traceRows"] == 42


def test_dpo_pairs_synthetic_warning() -> None:
    from tools.build_distill_dpo_pairs import build_pairs

    traces = [{"messages": [
        {"role": "user", "content": "Who wrote the Dao De Jing?"},
        {"role": "assistant", "content": "Laozi is traditionally credited; the text is compiled."},
    ], "metadata": {"taskId": "t1", "gatePassed": True}}]
    pairs, stats = build_pairs(traces, misses_by_prompt={})
    assert stats["pairs"] == 1 and stats["synthetic"] == 1 and stats["fromStudentMiss"] == 0
    assert pairs[0]["metadata"]["rejectedSource"] == "synthetic-template"
    assert stats["syntheticOnly"] is True


def test_dpo_pairs_from_real_miss() -> None:
    from tools.build_distill_dpo_pairs import build_pairs

    traces = [{"messages": [
        {"role": "user", "content": "Did Confucius write the Dao De Jing?"},
        {"role": "assistant", "content": "No — the Dao De Jing is Daoist, attributed to Laozi."},
    ], "metadata": {"taskId": "t2", "gatePassed": True}}]
    # a real base-student MISS the gate flags (forbidden attribution)
    misses = {"Did Confucius write the Dao De Jing?": "Yes, Confucius wrote the Dao De Jing."}
    pairs, stats = build_pairs(traces, misses)
    assert stats["fromStudentMiss"] == 1 and stats["synthetic"] == 0
    assert pairs[0]["metadata"]["rejectedSource"] == "student-miss"


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_distill_local: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
