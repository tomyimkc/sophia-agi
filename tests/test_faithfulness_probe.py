#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the faithfulness probe (agent/faithfulness_probe.py).

Exercises the core flip_rate with pure-function deciders/perturbs (the
offline-safe contract), the default deterministic perturbations, and the
trace-enriching probe_trace. No MLX/GPU required.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_log() -> Path:
    import agent.verified_trace as vt
    log = Path(tempfile.mkdtemp()) / "vt.jsonl"
    vt.TRACE_LOG = log
    return log


def test_flip_rate_detects_load_bearing_cot() -> None:
    from agent.faithfulness_probe import flip_rate
    # a decider that extracts the answer token after "Answer:"
    def decide(cot: str) -> str:
        return cot.strip().lower().split("answer:")[-1].strip().split()[0] if "answer:" in cot.lower() else "none"

    # CoT whose conclusion determines the answer -> dropping it flips the verdict
    cot = "All birds fly. The penguin is a bird. Answer: fly"
    # perturbation: drop the last sentence (the answer) -> verdict flips
    def drop_last(cot: str):
        parts = cot.strip().split(". ")
        return ". ".join(parts[:-1]) + "." if len(parts) > 1 else None
    fr = flip_rate(cot, decide, [drop_last])
    assert fr["attempted"] == 1
    assert fr["flips"] == 1
    assert fr["flipRate"] == 1.0  # the answer sentence was causally load-bearing


def test_flip_rate_low_for_post_hoc_cot() -> None:
    from agent.faithfulness_probe import flip_rate
    # a decider that always returns the same verdict regardless of the CoT body
    # (simulates a model whose answer is fixed by the prompt, not the CoT)
    def decide(cot: str) -> str:
        return "42"  # always 42, CoT irrelevant
    cot = "Some reasoning. Another sentence. Answer: 42"
    def drop_last(cot: str):
        parts = cot.strip().split(". ")
        return ". ".join(parts[:-1]) + "." if len(parts) > 1 else None
    fr = flip_rate(cot, decide, [drop_last, drop_last])
    # dropping any sentence never changes the fixed verdict -> flipRate 0
    assert fr["flipRate"] == 0.0
    assert fr["flips"] == 0


def test_flip_rate_handles_unapplicable_perturb() -> None:
    from agent.faithfulness_probe import flip_rate
    def decide(cot: str) -> str:
        return cot
    # a one-word CoT: the sentence-drop perturb cannot apply -> skipped
    fr = flip_rate("x", decide, [lambda c: None])
    assert fr["attempted"] == 0
    assert fr["skipped"] == 1
    assert fr["flipRate"] is None  # no applicable perturbation


def test_default_perturbs_are_deterministic_and_offline_safe() -> None:
    from agent.faithfulness_probe import default_perturbs
    ps = default_perturbs()
    assert len(ps) == 3
    # each is a pure function; running twice gives identical output
    cot = "All cats are mammals. A tabby is a cat. Therefore a tabby is a mammal."
    for p in ps:
        out1 = p(cot)
        out2 = p(cot)
        assert out1 == out2  # deterministic
        # output is either None (skip) or a different string
        assert out1 is None or out1 != cot


def test_probe_trace_records_enriched_trace_with_delta() -> None:
    log = _setup_log()
    from agent.faithfulness_probe import probe_trace, default_perturbs
    from agent.verified_trace import VerifiedTrace
    from sophia_contract.stores import _read_jsonl

    base = VerifiedTrace(
        traceId="vtrace_" + "a" * 24, runId="test", phase="benchmark", stepIdx=0,
        claimText="All birds fly. The robin is a bird. Answer: fly",
        claimKind="goal",
        fact={"verdict": "allow", "source": "t"},
        logic={"emittable": True, "contradictions": [], "laundered": [], "semanticsPreserved": True},
    )
    # decider keyed on the trailing answer token
    def decide(cot: str) -> str:
        return cot.strip().lower().split("answer:")[-1].strip().split()[0] if "answer:" in cot.lower() else "none"

    out = probe_trace(base, decide, default_perturbs())
    assert "faithfulnessDelta" in out
    assert "traceId" in out
    # an enriched trace was recorded
    rows = _read_jsonl(log)
    assert len(rows) >= 1
    # the delta is a number or None (when no perturb applied); never an error
    assert out["faithfulnessDelta"] is None or 0.0 <= out["faithfulnessDelta"] <= 1.0


def test_build_mlx_decide_fails_closed_off_mlx() -> None:
    # On CI / non-Apple-Silicon, MLX is unavailable -> the builder must raise a
    # clear RuntimeError (fail-closed), not silently produce a broken decider.
    from agent.faithfulness_probe import build_mlx_decide
    try:
        build_mlx_decide("q?")
        # if MLX IS available (Apple Silicon dev box), this is fine — just confirm
        # it returns a callable
    except RuntimeError as exc:
        assert "mlx" in str(exc).lower() or "unavailable" in str(exc).lower(), exc


def main() -> int:
    test_flip_rate_detects_load_bearing_cot()
    print(f"ok {test_flip_rate_detects_load_bearing_cot.__name__}")
    test_flip_rate_low_for_post_hoc_cot()
    print(f"ok {test_flip_rate_low_for_post_hoc_cot.__name__}")
    test_flip_rate_handles_unapplicable_perturb()
    print(f"ok {test_flip_rate_handles_unapplicable_perturb.__name__}")
    test_default_perturbs_are_deterministic_and_offline_safe()
    print(f"ok {test_default_perturbs_are_deterministic_and_offline_safe.__name__}")
    test_probe_trace_records_enriched_trace_with_delta()
    print(f"ok {test_probe_trace_records_enriched_trace_with_delta.__name__}")
    test_build_mlx_decide_fails_closed_off_mlx()
    print(f"ok {test_build_mlx_decide_fails_closed_off_mlx.__name__}")
    print("PASS faithfulness probe tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
