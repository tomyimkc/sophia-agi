# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for remediation executors + DCGM deep-diag parsing + manual heal path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster import ledger as L  # noqa: E402
from agent.cluster.executors import (  # noqa: E402
    KubeExecutor,
    NoopExecutor,
    SlurmExecutor,
    SSHExecutor,
    get_executor,
)
from agent.cluster.health import Verdict, evaluate_node  # noqa: E402
from agent.cluster.heal import execute_manual  # noqa: E402
from agent.cluster.ssh_provider import SSHTarget, parse_dcgm_diag, parse_probe  # noqa: E402


# --- DCGM deep diagnostic parsing -------------------------------------------
def _dcgm_json(statuses):
    return json.dumps({
        "DCGM GPU Diagnostic": {
            "test_categories": [
                {"category": "Hardware",
                 "tests": [{"name": name, "results": [{"status": st}]}
                           for name, st in statuses]}
            ]
        }
    })


def test_dcgm_all_pass_is_empty_tuple():
    raw = _dcgm_json([("GPU Memory", "Pass"), ("Diagnostic", "Pass")])
    assert parse_dcgm_diag(raw) == ()


def test_dcgm_failure_lists_failed_tests():
    raw = _dcgm_json([("GPU Memory", "Pass"), ("Diagnostic", "Fail")])
    assert parse_dcgm_diag(raw) == ("Diagnostic",)


def test_dcgm_absent_or_garbage_is_none():
    assert parse_dcgm_diag("") is None
    assert parse_dcgm_diag("not json") is None
    assert parse_dcgm_diag("{}") is None


def test_dcgm_failure_makes_node_fail():
    raw = "===NVIDIA_SMI===\nNVIDIA H100, 60, 50, 0, 0x0\n===END===\n===DCGM===\n" \
          + _dcgm_json([("Diagnostic", "Fail")]) + "\n===DCGM_END===\n"
    m = parse_probe("n", raw)
    assert m.dcgm_diag == ("Diagnostic",)
    h = evaluate_node(m)
    assert h.verdict == Verdict.FAIL
    assert any(r.signal == "dcgm_diag" for r in h.reasons)


def test_dcgm_not_run_does_not_penalize():
    # No DCGM section → dcgm_diag None → opt-in check absent, no FAIL from it.
    m = parse_probe("n", "===NVIDIA_SMI===\nNVIDIA H100, 60, 50, 0, 0x0\n===END===\n")
    assert m.dcgm_diag is None


# --- Executors (injectable runner; no real cluster) -------------------------
def _capturing_runner(calls):
    def run(argv):
        calls.append(argv)
        return 0, "ok"
    return run


def test_kube_cordon_and_drain_commands():
    calls = []
    ex = KubeExecutor(runner=_capturing_runner(calls))
    assert ex("node-1", "cordon_and_investigate") is True
    assert calls[-1] == ["kubectl", "cordon", "node-1"]
    assert ex("node-1", "drain_and_reboot") is True
    assert calls[-1][:3] == ["kubectl", "drain", "node-1"]
    assert "--ignore-daemonsets" in calls[-1]


def test_kube_unsupported_action_returns_false():
    ex = KubeExecutor(runner=_capturing_runner([]))
    assert ex("node-1", "gc_disk") is False  # node-local fix isn't kube's job


def test_kube_node_name_mapping():
    calls = []
    ex = KubeExecutor(runner=_capturing_runner(calls), node_name=lambda nid: f"k8s-{nid}")
    ex("pod-9", "cordon_and_investigate")
    assert calls[-1] == ["kubectl", "cordon", "k8s-pod-9"]


def test_slurm_drain_and_reboot_emits_two_commands():
    calls = []
    ex = SlurmExecutor(runner=_capturing_runner(calls))
    assert ex("gpu07", "drain_and_reboot") is True
    assert calls[0][:2] == ["scontrol", "update"]
    assert "state=drain" in calls[0]
    assert calls[1] == ["scontrol", "reboot", "gpu07"]


def test_runner_failure_propagates():
    def failing(argv):
        return 1, "boom"
    ex = KubeExecutor(runner=failing)
    assert ex("n", "cordon_and_investigate") is False


def test_ssh_executor_builds_remote_command():
    calls = []
    ex = SSHExecutor(runner=_capturing_runner(calls), key_path="/tmp/key",
                     targets={"n1": SSHTarget("n1", "10.0.0.1", 2200)})
    assert ex("n1", "restore_telemetry") is True
    argv = calls[-1]
    assert argv[0] == "ssh" and "root@10.0.0.1" in argv
    assert "nv-hostengine" in argv[-1]   # the remote command body


def test_ssh_executor_rejects_scheduler_action():
    ex = SSHExecutor(runner=_capturing_runner([]), key_path="/tmp/key")
    assert ex("n1", "drain_and_reboot") is False  # not a node-local fix


def test_noop_simulates_success():
    assert NoopExecutor()("n", "drain_and_reboot") is True


def test_dry_run_does_not_call_runner():
    calls = []
    ex = KubeExecutor(runner=_capturing_runner(calls), dry_run=True)
    assert ex("n", "cordon_and_investigate") is True
    assert calls == []  # nothing actually ran


def test_get_executor_factory():
    assert isinstance(get_executor("kube"), KubeExecutor)
    try:
        get_executor("bogus")
        assert False
    except ValueError:
        pass


# --- Manual (human-approved) heal path --------------------------------------
def test_manual_requires_approval(tmp_path):
    audit = tmp_path / "a.jsonl"
    calls = []
    ex = KubeExecutor(runner=_capturing_runner(calls))
    ok = execute_manual("n", "drain_and_reboot", ex, approved=False, audit_path=audit)
    assert ok is False
    assert calls == []  # never executed without approval
    rec = json.loads(audit.read_text().splitlines()[-1])
    assert rec["executed"] is False


def test_manual_approved_executes_and_records_recovery(tmp_path):
    audit = tmp_path / "a.jsonl"
    led = tmp_path / "incidents.jsonl"
    iid = L.record_detection(node_id="n", kind="xid_errors", severity="FAIL",
                             detected_at="2026-06-26T00:00:00+00:00", ledger=led)
    calls = []
    ex = KubeExecutor(runner=_capturing_runner(calls))
    ok = execute_manual("n", "drain_and_reboot", ex, operator="alice",
                        approved=True, audit_path=audit, incident_id=iid, ledger=led)
    assert ok is True
    assert calls and calls[-1][:2] == ["kubectl", "drain"]
    stats = L.mttr_stats(led)
    assert stats["recovered"] == 1
    assert stats["auto_healed"] == 0  # a human resolved it, not auto-heal
