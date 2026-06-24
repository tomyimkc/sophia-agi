#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the 1.1.0 guardrails & memory: capability scopes per role, dry-run,
kill switch, durable idempotent task queue, Langfuse-compatible traces, and the
per-verdict ROI estimate. Deterministic, offline.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_contract import Scope, ScopeRegistry, SophiaContract  # noqa: E402

_CLK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731


def _svc(**kw):
    return SophiaContract(clock=_CLK, **kw)


def test_scope_denies_unscoped_op() -> None:
    svc = _svc(scopes=ScopeRegistry({"reader": Scope(ops={"verify_claim"}, max_blp="UNCLASSIFIED")}))
    out = svc.record_claim({"idempotency_key": "k", "content": "x", "sources": ["s"], "role": "reader"})
    assert out["error"]["code"] == "UNAUTHENTICATED"


def test_scope_unknown_role_fails_closed() -> None:
    svc = _svc(scopes=ScopeRegistry({"reader": Scope(ops={"verify_claim"})}))
    out = svc.verify_claim({"claim_id": "clm_x", "role": "ghost"})
    assert out["error"]["code"] == "UNAUTHENTICATED"


def test_scope_blp_cap_enforced() -> None:
    svc = _svc(scopes=ScopeRegistry({"researcher": Scope(ops={"record_claim"}, max_blp="CONFIDENTIAL")}))
    out = svc.record_claim({"idempotency_key": "k", "content": "x", "sources": ["s"],
                            "blp_level": "SECRET", "role": "researcher"})
    assert out["error"]["code"] == "UNAUTHENTICATED"


def test_no_registry_is_unrestricted() -> None:
    svc = _svc()  # empty registry
    out = svc.record_claim({"idempotency_key": "k", "content": "x", "sources": ["s"], "role": "anyone"})
    assert "error" not in out


def test_dry_run_does_not_persist() -> None:
    svc = _svc()
    dr = svc.record_claim({"idempotency_key": "d", "content": "draft", "sources": ["s"], "dry_run": True})
    assert dr.get("dry_run") is True
    assert svc.claims.get_by_id(dr["claim_id"]) is None  # not stored


def test_kill_switch_halts_and_recovers() -> None:
    svc = _svc()
    svc.engage_kill_switch("incident")
    rec = svc.record_claim({"idempotency_key": "k", "content": "x", "sources": ["s"]})
    assert rec["error"]["code"] == "UNAVAILABLE" and rec["error"]["retryable"] is True
    assert svc.health()["status"] == "degraded"
    svc.release_kill_switch()
    assert "error" not in svc.record_claim({"idempotency_key": "k", "content": "x", "sources": ["s"]})
    assert svc.health()["status"] == "ok"


def test_task_queue_idempotent_and_durable() -> None:
    d = Path(tempfile.mkdtemp())
    svc = SophiaContract(store_dir=d, clock=_CLK)
    t1 = svc.enqueue_task({"idempotency_key": "job", "kind": "verify", "payload": {"x": 1}})
    t2 = svc.enqueue_task({"idempotency_key": "job", "kind": "verify"})
    assert t1["task_id"] == t2["task_id"]            # idempotent
    leased = svc.next_task()["task"]
    assert leased["task_id"] == t1["task_id"] and leased["state"] == "leased"
    svc.complete_task(t1["task_id"], result={"ok": True})
    # durable: a fresh instance sees the completed task and no pending work
    svc2 = SophiaContract(store_dir=d, clock=_CLK)
    assert svc2.task_status(t1["task_id"])["state"] == "done"
    assert svc2.tasks.pending_count() == 0


def test_traces_langfuse_shape() -> None:
    svc = _svc()
    c = svc.record_claim({"idempotency_key": "t", "content": "x", "sources": ["s"]})
    svc.verify_claim({"claim_id": c["claim_id"]})
    events = svc.trace()["events"]
    assert len(events) >= 2
    for e in events:
        assert {"id", "name", "startTime", "endTime", "input", "output", "level", "metadata"} <= set(e)


def test_roi_estimate_present_and_sane() -> None:
    svc = _svc()
    c = svc.record_claim({"idempotency_key": "roi", "content": "x", "sources": ["s1", "s2"]})
    accepted = svc.verify_claim({"claim_id": c["claim_id"]})
    assert accepted["verdict"] == "accepted"
    assert accepted["roi_estimate"]["founder_minutes_saved"] > 0
    held = svc.verify_claim({"claim_id": svc.record_claim(
        {"idempotency_key": "roi2", "content": "y", "sources": []})["claim_id"]})
    assert held["verdict"] == "held"
    assert held["roi_estimate"]["founder_minutes_saved"] == 0.0


def test_describe_advertises_new_capabilities() -> None:
    caps = SophiaContract().describe()["capabilities"]
    for c in ("enqueue_task", "next_task", "trace", "health", "explain_verdict", "batch_verify"):
        assert c in caps


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_contract_guardrails: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
