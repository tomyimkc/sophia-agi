#!/usr/bin/env python3
"""Offline tests for WS-D: no-executor ablation flag + timed long-horizon runner."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


# ---------------- timed long-horizon (fail-closed + event log) ----------------
def _load_timed_with_stubs():
    lh = types.ModuleType("agent.long_horizon")

    class _Result:
        def __init__(self):
            self.ledger_id = "L1"; self.ok = True
            self.completed = ["n1"]; self.failed = []; self.blocked = []
            self.total_cost_usd = 0.0; self.ledger_path = "/tmp/ledger.json"

    class _Ledger:
        ledger_id = "L1"

    lh.build_ledger = lambda goal, subtasks, ledger_id: _Ledger()
    lh.run_long_horizon = lambda ledger, client=None, approve_tools=False, **kw: _Result()

    # snapshot so we can restore the REAL agent.* after the tool captures its refs —
    # otherwise the empty-path 'agent' stub shadows the real package for sibling suites
    _keys = ("agent", "agent.long_horizon")
    _saved = {k: sys.modules.get(k) for k in _keys}
    agent_pkg = types.ModuleType("agent"); agent_pkg.__path__ = []
    sys.modules.update({"agent": agent_pkg, "agent.long_horizon": lh})
    spec = importlib.util.spec_from_file_location("timedtool", TOOLS / "run_long_horizon_timed.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for k, v in _saved.items():  # restore real modules (or remove stub if none was present)
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    return mod


def test_event_log_append_only(tmp_path):
    m = _load_timed_with_stubs()
    log = m.EventLog(tmp_path / "e.jsonl")
    log.emit("run_start", budget_minutes=30)
    log.emit("human_intervention", node="n1")
    log.emit("model_call", prompt_chars=10)
    log.close()
    lines = (tmp_path / "e.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert log.interventions == 1
    assert all("ts" in json.loads(l) for l in lines)


def test_env_artifact_no_backend(tmp_path):
    m = _load_timed_with_stubs()
    out = tmp_path / "r.json"
    m.env_artifact("no backend", out)
    rep = json.loads(out.read_text())
    assert rep["environmentArtifact"] and rep["completedRun"] is False
    assert rep["canClaimAGI"] is False


def test_no_backend_client_is_none(tmp_path):
    m = _load_timed_with_stubs()
    log = m.EventLog(tmp_path / "e.jsonl")
    # The mock provider (auto-selected when no API key) is treated as NO backend: the
    # D1 guard must fail-closed to None. Force the mock spec so this is deterministic
    # regardless of what real backend the host environment happens to have configured
    # (e.g. a grok CLI auth file makes resolve_config(None) resolve to a real client).
    assert m._make_client("mock", log) is None
    log.close()


# ---------------- ablation flag presence (against the real fetched source) ----
def test_patch_documents_use_executor_flag():
    """The patch file must add use_executor to Ablation and a sophia-no-executor mode."""
    patch = (TOOLS / "ablation_no_executor.patch.md").read_text()
    assert "use_executor: bool = True" in patch
    assert "sophia-no-executor" in patch
    assert "if ablation.use_executor" in patch


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
