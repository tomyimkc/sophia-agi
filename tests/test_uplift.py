#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the harness-uplift benchmark (offline, deterministic)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402
from agent import uplift as up  # noqa: E402

_GOOD = "[ok] Analysis.\nDecision: proceed. source discipline noted.\n中文摘要: 完成。"

_SUITE = [
    {"id": "c1", "goal": "should we ship this week", "mode": "advisor"},
    {"id": "c2", "goal": "what to validate before release", "mode": "advisor"},
]


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


class _HarnessOnlyClient:
    """Passes only inside the harness step prompt (which carries '## Current step');
    the bare single-shot prompt lacks that marker and gets an empty (failing) answer.
    So the harness loop demonstrably lifts a model that fails bare."""

    def generate(self, system: str, user: str):
        if "## Current step" in user:
            return m.ModelResult(text=_GOOD, provider="stub", model="stub", ok=True)
        return m.ModelResult(text="", provider="stub", model="stub", ok=True)


def test_positive_uplift_when_only_harness_passes() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        res = up.measure_uplift(_SUITE, client=_HarnessOnlyClient(), max_retries=1, bootstrap_seed=1)
        assert res.bare_pass_rate == 0.0
        assert res.harness_pass_rate == 1.0
        assert res.uplift == 1.0
        # All paired deltas are +1 -> bootstrap CI is degenerate at 1.0 and demonstrated.
        assert res.uplift_ci95 == (1.0, 1.0)
        assert res.demonstrated is True
        assert [c.delta for c in res.cases] == [1, 1]


def test_no_uplift_when_bare_already_passes() -> None:
    # The mock model already produces a gate-friendly answer bare, so the harness
    # adds nothing measurable -> uplift 0, not demonstrated.
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        res = up.measure_uplift(_SUITE, client=_mock_client(), max_retries=1, bootstrap_seed=1)
        assert res.bare_pass_rate == 1.0
        assert res.harness_pass_rate == 1.0
        assert res.uplift == 0.0
        assert res.demonstrated is False


def test_bare_answer_is_single_shot_text() -> None:
    task = h.AgentTask(goal="x", mode="advisor")
    text = up.bare_answer(task, client=_mock_client())
    assert isinstance(text, str) and text.strip()


def test_bootstrap_ci_is_seeded_and_ordered() -> None:
    deltas = [1, 0, 1, 0, 1, -1, 1, 0]
    a = up._bootstrap_ci(deltas, seed=7)
    b = up._bootstrap_ci(deltas, seed=7)
    assert a == b  # deterministic for a fixed seed
    assert a[0] <= sum(deltas) / len(deltas) <= a[1]  # point estimate inside the CI
    # Degenerate cases.
    assert up._bootstrap_ci([], seed=0) == (0.0, 0.0)
    assert up._bootstrap_ci([1], seed=0) == (1.0, 1.0)


def test_result_to_dict_shape() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        res = up.measure_uplift(_SUITE[:1], client=_mock_client(), max_retries=0)
        d = res.to_dict()
    for key in ("suiteSize", "barePassRate", "harnessPassRate", "uplift", "upliftCi95", "demonstrated", "cases"):
        assert key in d
    assert isinstance(d["upliftCi95"], list) and len(d["upliftCi95"]) == 2


def main() -> int:
    test_positive_uplift_when_only_harness_passes()
    test_no_uplift_when_bare_already_passes()
    test_bare_answer_is_single_shot_text()
    test_bootstrap_ci_is_seeded_and_ordered()
    test_result_to_dict_shape()
    print("test_uplift: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
