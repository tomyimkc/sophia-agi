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

from tools.runpod_connect import (  # noqa: E402
    DISPATCH_WORKFLOW, classify_pod, main, resolve_api_key,
)


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
