# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Harder SSIL micro-task — string transformation (ARC-like, executable spec).

Demonstrates the loop generalizes beyond a single threshold task. The model proposes
an executable TRANSFORM SPEC (an ordered pipeline of safe string ops); we run it on
held-out (input -> gold) pairs the model never saw. Exact-match accuracy feeds G4;
the protected metric guards against a degenerate identity/empty transform.

Ops are a tiny whitelisted set (no eval, no arbitrary code): upper, lower, reverse,
strip, title, and `replace:a:b`. The latent gold transform is
``strip -> upper -> replace ' ' with '_'``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, "") and str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.continual_plasticity import EvalMetric

_SAFE_OPS = {"upper", "lower", "reverse", "strip", "title"}


def _apply_op(s: str, op: str) -> str:
    if op == "upper":
        return s.upper()
    if op == "lower":
        return s.lower()
    if op == "reverse":
        return s[::-1]
    if op == "strip":
        return s.strip()
    if op == "title":
        return s.title()
    if op.startswith("replace:"):
        parts = op.split(":", 2)
        if len(parts) == 3:
            return s.replace(parts[1], parts[2])
    return s  # unknown op = no-op (safe)


@dataclass(frozen=True)
class TransformSpec:
    ops: tuple[str, ...]

    def apply(self, s: str) -> str:
        for op in self.ops:
            s = _apply_op(s, op)
        return s

    @classmethod
    def from_list(cls, ops: list[str]) -> "TransformSpec":
        clean = [o for o in ops if isinstance(o, str) and (o in _SAFE_OPS or o.startswith("replace:"))]
        return cls(ops=tuple(clean))


def _gold(s: str) -> str:
    return s.strip().upper().replace(" ", "_")


def _cases() -> list[dict]:
    raw_train = ["  hello world", "Foo Bar", "a b c"]
    raw_test = ["  mixed Case here ", "one two three", "Sophia agi loop", "  trailing  "]
    train = [{"input": r, "gold": _gold(r), "split": "train"} for r in raw_train]
    test = [{"input": r, "gold": _gold(r), "split": "test"} for r in raw_test]
    return train + test


def load_cases(split: str | None = None) -> list[dict]:
    return [c for c in _cases() if split is None or c["split"] == split]


def baseline_spec() -> TransformSpec:
    return TransformSpec(ops=())  # identity


def _accuracy(spec: TransformSpec, cases: list[dict]) -> float:
    if not cases:
        return 0.0
    return round(sum(spec.apply(c["input"]) == c["gold"] for c in cases) / len(cases), 4)


def _nonidentity_rate(spec: TransformSpec, cases: list[dict]) -> float:
    """Protected: fraction of cases the transform actually changes. Identity = 0."""
    if not cases:
        return 1.0
    return round(sum(spec.apply(c["input"]) != c["input"] for c in cases) / len(cases), 4)


NONIDENTITY_FLOOR = 0.5


def measure_transform(candidate: TransformSpec, *, baseline: "TransformSpec | None" = None) -> tuple[tuple[EvalMetric, ...], dict[str, Any]]:
    test = load_cases("test")
    base = baseline or baseline_spec()
    metrics = (
        EvalMetric("transform_accuracy", _accuracy(base, test), _accuracy(candidate, test), protected=False),
        EvalMetric("nonidentity_rate", NONIDENTITY_FLOOR, _nonidentity_rate(candidate, test), protected=True),
    )
    detail = {
        "testCases": len(test), "nonidentityFloor": NONIDENTITY_FLOOR,
        "baseline": {"accuracy": _accuracy(base, test)},
        "candidate": {"accuracy": _accuracy(candidate, test), "ops": list(candidate.ops)},
    }
    return metrics, detail


def train_feature_summary() -> dict[str, Any]:
    train = load_cases("train")
    return {
        "task": "Propose an ordered pipeline of string ops mapping input -> output.",
        "allowedOps": sorted(_SAFE_OPS) + ["replace:<from>:<to>"],
        "examples": [{"input": c["input"], "output": c["gold"]} for c in train],
    }


def demo_transform_report() -> dict[str, Any]:
    good = TransformSpec.from_list(["strip", "upper", "replace: :_"])
    identity = baseline_spec()
    test = load_cases("test")
    return {
        "schema": "sophia.ssil_transform_demo.v1", "candidateOnly": True, "level3Evidence": False,
        "goodAccuracy": _accuracy(good, test), "identityAccuracy": _accuracy(identity, test),
        "invariants": {
            "good_solves_task": _accuracy(good, test) == 1.0,
            "identity_fails": _accuracy(identity, test) == 0.0,
            "identity_tanks_protected": _nonidentity_rate(identity, test) == 0.0,
        },
    }


if __name__ == "__main__":
    print(json.dumps(demo_transform_report(), ensure_ascii=False, indent=2))
