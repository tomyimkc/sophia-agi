#!/usr/bin/env python3
"""D4: run_long_horizon's cooperative wall-clock deadline actually stops a multi-node run.

Review defect D4 was that tools/run_long_horizon_timed.py's --minutes budget was a
post-hoc marker only — run_long_horizon had no deadline hook, so a runaway tree could
never be interrupted. These tests pin the fix: a `deadline_monotonic` that stops the
engine at node granularity (never launching a node past the budget). run_subagent is
stubbed so no backend is touched; a fake monotonic clock makes timing deterministic.
"""
from __future__ import annotations

import agent.long_horizon as lh


class _Child:
    """Minimal stand-in for SubagentResult (the fields run_long_horizon reads)."""
    ok = True
    cost_usd = 0.0
    final_text = "done"
    failures: list[str] = []


def _ledger(tmp_path, n=3):
    subtasks = [{"id": f"n{i}", "goal": f"do {i}"} for i in range(1, n + 1)]
    return lh.build_ledger("root goal", subtasks, ledger_id="deadline-test", ledgers_dir=tmp_path)


def test_expired_deadline_runs_zero_nodes(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _stub(*a, **k):
        calls["n"] += 1
        return _Child()

    monkeypatch.setattr(lh, "run_subagent", _stub)
    led = _ledger(tmp_path, 3)
    # deadline already in the past -> the loop must break before launching any node.
    res = lh.run_long_horizon(led, client=object(), deadline_monotonic=lh.time.monotonic() - 1.0)
    assert calls["n"] == 0, "a node was launched past an already-expired budget"
    assert res.completed == []
    assert res.ok is False


def test_deadline_stops_partway(monkeypatch, tmp_path):
    calls = {"n": 0}

    def _stub(*a, **k):
        calls["n"] += 1
        return _Child()

    monkeypatch.setattr(lh, "run_subagent", _stub)
    # Fake clock: 1st check 0.0 (< deadline 1.0) -> node runs; 2nd check 100.0 -> break.
    ticks = iter([0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(lh.time, "monotonic", lambda: next(ticks))
    led = _ledger(tmp_path, 3)
    res = lh.run_long_horizon(led, client=object(), deadline_monotonic=1.0)
    assert calls["n"] == 1, "budget did not stop the run after the first node"
    assert len(res.completed) == 1
    assert res.ok is False  # 1 of 3 nodes done -> not a complete tree


def test_no_deadline_runs_all_nodes(monkeypatch, tmp_path):
    """Regression: deadline_monotonic=None (default) preserves prior unbounded behavior."""
    calls = {"n": 0}

    def _stub(*a, **k):
        calls["n"] += 1
        return _Child()

    monkeypatch.setattr(lh, "run_subagent", _stub)
    led = _ledger(tmp_path, 3)
    res = lh.run_long_horizon(led, client=object())
    assert calls["n"] == 3
    assert len(res.completed) == 3
    assert res.ok is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
