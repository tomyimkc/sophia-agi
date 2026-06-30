#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A2→A3 bridge: pairwise verdicts -> content-uplift judgments schema.

Deterministic, offline. Verifies the documented encoding and the abstention drop-rule
that keeps the assembled schema consistent with run_lora_uplift_validation's family κ.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.assemble_uplift_judgments import _content_pair, assemble  # noqa: E402
from tools.run_lora_uplift_validation import _family_key  # noqa: E402


def test_pairwise_encoding() -> None:
    assert _content_pair("adapter") == (True, False)
    assert _content_pair("base") == (False, True)
    assert _content_pair("tie") == (True, True)
    assert _content_pair(None) is None
    assert _content_pair("garbage") is None


def test_assemble_drops_abstained_family_not_whole_item() -> None:
    raws = [{
        "seed": 0,
        "judges": ["ollama:qwen", "vllm:mlx/Llama-70B"],
        "answers": "seed0.json",
        "items": [
            {"id": "a", "verdicts": {"ollama:qwen": "adapter", "vllm:mlx/Llama-70B": "adapter"}},
            {"id": "b", "verdicts": {"ollama:qwen": "base", "vllm:mlx/Llama-70B": None}},
        ],
    }]
    j = assemble(raws, "allenai/OLMoE-1B-7B-0924-Instruct")
    assert j["subjectModel"] == "allenai/OLMoE-1B-7B-0924-Instruct"
    items = {it["id"]: it for it in j["seeds"][0]["items"]}
    # CONTENT must be keyed by the gate's family key, NOT the raw spec (the 2026-06-30 bug).
    fq, fm = _family_key("ollama:qwen"), _family_key("vllm:mlx/Llama-70B")
    assert fq != fm and fq == "ollama"
    # item a: both families labelled, keyed by family
    assert items["a"]["adapterContent"] == {fq: True, fm: True}
    assert "ollama:qwen" not in items["a"]["adapterContent"]   # NOT the raw spec
    # item b: 70B abstained -> only qwen family present (consistent with A3 all-families drop rule)
    assert set(items["b"]["adapterContent"]) == {fq}
    assert fm not in items["b"]["baseContent"]


def test_same_family_collision_refused() -> None:
    # two specs that map to the SAME family key must be refused (the gate would collapse them)
    raws = [{"seed": 0, "judges": ["ollama:qwen2.5:7b", "ollama:qwen2.5:32b"], "answers": "s",
             "items": [{"id": "a", "verdicts": {"ollama:qwen2.5:7b": "adapter",
                                                "ollama:qwen2.5:32b": "base"}}]}]
    try:
        assemble(raws, "allenai/OLMoE")
        raised = False
    except ValueError:
        raised = True
    assert raised, "two specs in the same family must be refused"


def test_refuses_mixed_judge_sets() -> None:
    raws = [
        {"seed": 0, "judges": ["a", "b"], "items": []},
        {"seed": 1, "judges": ["a", "c"], "items": []},  # drift
    ]
    try:
        assemble(raws, "allenai/OLMoE")
        raised = False
    except ValueError:
        raised = True
    assert raised, "a drifting judge set across seeds must be refused (would corrupt κ)"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} assemble_uplift_judgments tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
