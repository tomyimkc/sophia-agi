#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for tools/runpod_connect.py (no network).

Cover the two things the connect skill promises: (1) honest key resolution with a
GitHub-mediated fallback when absent, and (2) the REST-only stall classifier.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.runpod_connect as rc  # noqa: E402
from tools.runpod_connect import (  # noqa: E402
    DISPATCH_WORKFLOW, RunPodError, classify_pod, main, reap_exited,
    resolve_api_key, terminate_pod,
)

_LEAKED_FLEET = [
    {"id": "exit1", "name": "sophia-rlvr-20260626-111704", "desiredStatus": "EXITED"},
    {"id": "run1", "name": "live", "desiredStatus": "RUNNING",
     "runtime": {"uptimeInSeconds": 50}},
]


# --- key resolution -------------------------------------------------------

def test_resolve_prefers_explicit_arg() -> None:
    key, source = resolve_api_key("rpa_explicit", env={"RUNPOD_API_KEY": "rpa_env"})
    assert key == "rpa_explicit"
    assert source == "arg"


def test_resolve_from_env() -> None:
    key, source = resolve_api_key(None, env={"RUNPOD_API_KEY": "rpa_env"})
    assert (key, source) == ("rpa_env", "env")


def test_resolve_missing_is_honest() -> None:
    key, source = resolve_api_key(None, env={})
    assert key is None
    assert source == "missing"


def test_resolve_blank_env_is_missing() -> None:
    key, source = resolve_api_key(None, env={"RUNPOD_API_KEY": "   "})
    assert key is None and source == "missing"


# --- stall classification (REST-only heuristic) ---------------------------

def test_running_with_uptime_is_healthy() -> None:
    v = classify_pod({"id": "p1", "desiredStatus": "RUNNING",
                      "runtime": {"uptimeInSeconds": 1200}})
    assert v["verdict"] == "running"


def test_running_without_runtime_is_stalled() -> None:
    v = classify_pod({"id": "p2", "name": "trainer", "desiredStatus": "RUNNING",
                      "runtime": None})
    assert v["verdict"] == "stalled"
    assert v["id"] == "p2" and v["name"] == "trainer"


def test_running_with_zero_uptime_is_stalled() -> None:
    v = classify_pod({"id": "p3", "desiredStatus": "RUNNING",
                      "runtime": {"uptimeInSeconds": 0}})
    assert v["verdict"] == "stalled"


def test_exited_is_stopped() -> None:
    assert classify_pod({"id": "p4", "desiredStatus": "EXITED"})["verdict"] == "stopped"


def test_unknown_status() -> None:
    assert classify_pod({"id": "p5", "desiredStatus": ""})["verdict"] == "unknown"


# --- CLI dry-run picks the right route, no network ------------------------

def test_dry_run_direct_route_when_key_present(monkeypatch, capsys) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_present")
    rc = main(["--dry-run", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"route": "direct"' in out


def test_dry_run_github_fallback_when_key_absent(monkeypatch, capsys) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    rc = main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert DISPATCH_WORKFLOW in out  # points at the GitHub-mediated route


def test_check_without_key_fails_closed(monkeypatch, capsys) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    rc = main(["--check", "--json"])
    out = capsys.readouterr().out
    assert rc == 2
    assert '"connected": false' in out
    assert DISPATCH_WORKFLOW in out


# --- terminate (cost-saving) path -----------------------------------------

def test_terminate_pod_success(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(rc, "_api_request",
                        lambda m, p, k, *a, **kw: calls.append((m, p)) or None)
    res = terminate_pod("rpa_x", "111704")
    assert res == {"id": "111704", "terminated": True, "detail": "deleted"}
    assert calls == [("DELETE", "/pods/111704")]


def test_terminate_pod_404_is_success(monkeypatch) -> None:
    def boom(*a, **kw):
        raise RunPodError("RunPod API DELETE /pods/x failed: HTTP 404: gone")
    monkeypatch.setattr(rc, "_api_request", boom)
    res = terminate_pod("rpa_x", "111704")
    assert res["terminated"] is True and "404" in res["detail"]


def test_terminate_pod_error_is_failure(monkeypatch) -> None:
    def boom(*a, **kw):
        raise RunPodError("HTTP 500: boom")
    monkeypatch.setattr(rc, "_api_request", boom)
    res = terminate_pod("rpa_x", "111704")
    assert res["terminated"] is False


def test_cli_terminate_requires_yes(monkeypatch, capsys) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_present")
    monkeypatch.setattr(rc, "get_pod", lambda k, pid: {"id": pid, "desiredStatus": "RUNNING",
                                                       "runtime": {"uptimeInSeconds": 99}})
    deleted = []
    monkeypatch.setattr(rc, "terminate_pod", lambda k, pid: deleted.append(pid))
    rc_code = main(["--terminate", "111704", "--json"])
    out = capsys.readouterr().out
    assert rc_code == 2  # blocked without --yes
    assert deleted == []  # nothing deleted
    assert "needs --yes" in out


def test_cli_terminate_with_yes_deletes(monkeypatch, capsys) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_present")
    monkeypatch.setattr(rc, "get_pod", lambda k, pid: {"id": pid, "desiredStatus": "RUNNING",
                                                       "runtime": {"uptimeInSeconds": 99}})
    monkeypatch.setattr(rc, "terminate_pod",
                        lambda k, pid: {"id": pid, "terminated": True, "detail": "deleted"})
    rc_code = main(["--terminate", "111704", "--yes", "--json"])
    out = capsys.readouterr().out
    assert rc_code == 0
    assert '"terminated": true' in out


# --- reaper (leaked EXITED-pod cleanup) ------------------------------------

def test_reap_preview_does_not_delete(monkeypatch) -> None:
    monkeypatch.setattr(rc, "_list_pods", lambda k: list(_LEAKED_FLEET))
    deleted = []
    monkeypatch.setattr(rc, "terminate_pod", lambda k, pid: deleted.append(pid))
    res = reap_exited("rpa_x", do_delete=False)
    assert res["leaked_count"] == 1
    assert res["leaked"][0]["id"] == "exit1"  # only the EXITED pod, not the running one
    assert deleted == []  # preview never deletes


def test_reap_with_delete_terminates_only_exited(monkeypatch) -> None:
    monkeypatch.setattr(rc, "_list_pods", lambda k: list(_LEAKED_FLEET))
    deleted = []
    monkeypatch.setattr(rc, "terminate_pod",
                        lambda k, pid: deleted.append(pid) or {"id": pid, "terminated": True,
                                                               "detail": "deleted"})
    res = reap_exited("rpa_x", do_delete=True)
    assert deleted == ["exit1"]  # the running pod is left alone
    assert res["actions"][0]["terminated"] is True


def test_cli_reap_preview_exit_code_3(monkeypatch, capsys) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_present")
    monkeypatch.setattr(rc, "_list_pods", lambda k: list(_LEAKED_FLEET))
    rc_code = main(["--reap-exited", "--json"])
    out = capsys.readouterr().out
    assert rc_code == 3  # leaks exist, preview-only → non-zero so callers notice
    assert '"leaked_count": 1' in out


if __name__ == "__main__":  # allow running without pytest installed
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        if fn.__code__.co_argcount:  # needs pytest fixtures; skip in bare mode
            continue
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    raise SystemExit(1 if failed else 0)
