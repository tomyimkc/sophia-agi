# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the fail-closed node health evaluator and fault-localization playbook."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster.health import NodeMetrics, Thresholds, Verdict, evaluate_node  # noqa: E402
from agent.cluster.playbook import Risk, diagnose, primary_diagnosis  # noqa: E402
from agent.cluster.provider import MockProvider, sweep  # noqa: E402


def _healthy(**over) -> NodeMetrics:
    base = dict(node_id="n0", gpu_temp_c=60.0, gpu_util=0.8, mem_used_frac=0.5,
                disk_used_frac=0.4, ecc_uncorrectable=0, nvlink_down=0,
                rdma_link_down=0, throttled=False)
    base.update(over)
    return NodeMetrics(**base)


def test_healthy_node_passes():
    h = evaluate_node(_healthy())
    assert h.verdict == Verdict.PASS
    assert h.ok and not h.failed
    assert h.reasons == []


def test_unreachable_is_fail_and_short_circuits():
    h = evaluate_node(NodeMetrics(node_id="n", reachable=False))
    assert h.verdict == Verdict.FAIL
    assert h.reasons[0].signal == "reachability"
    assert len(h.reasons) == 1  # short-circuits, no spurious unknown-metric warns


def test_missing_critical_metric_is_warn_not_pass():
    # Reachable but temperature unknown → fail-closed WARN, never PASS.
    h = evaluate_node(NodeMetrics(node_id="n", reachable=True, ecc_uncorrectable=0,
                                  nvlink_down=0, rdma_link_down=0))
    assert h.verdict == Verdict.WARN
    assert any(r.signal == "gpu_temp_c" and r.verdict == Verdict.WARN for r in h.reasons)


def test_fatal_xid_79_fails():
    h = evaluate_node(_healthy(xid_errors=(79,)))
    assert h.verdict == Verdict.FAIL
    assert any(r.signal == "xid_errors" and r.verdict == Verdict.FAIL for r in h.reasons)


def test_uncorrectable_ecc_fails():
    h = evaluate_node(_healthy(ecc_uncorrectable=2))
    assert h.failed


def test_rdma_and_nvlink_down_fail():
    assert evaluate_node(_healthy(rdma_link_down=1)).failed
    assert evaluate_node(_healthy(nvlink_down=1)).failed


def test_temp_warn_vs_fail_thresholds():
    assert evaluate_node(_healthy(gpu_temp_c=83.0)).verdict == Verdict.WARN
    assert evaluate_node(_healthy(gpu_temp_c=90.0)).verdict == Verdict.FAIL


def test_custom_thresholds_respected():
    t = Thresholds(temp_warn_c=50.0, temp_fail_c=55.0)
    assert evaluate_node(_healthy(gpu_temp_c=52.0), t).verdict == Verdict.WARN


def test_diagnosis_cites_signal_and_sets_risk():
    h = evaluate_node(_healthy(xid_errors=(79,)))
    diags = diagnose(h)
    assert diags and diags[0].signal == "xid_errors"
    assert diags[0].action == "drain_and_reboot"
    assert diags[0].confidence >= 0.85
    primary = primary_diagnosis(h)
    assert primary is not None and primary.risk in (Risk.LOW, Risk.MEDIUM, Risk.HIGH)


def test_primary_diagnosis_prefers_worst_risk():
    # ECC (HIGH risk) should outrank a co-occurring disk warning (LOW).
    h = evaluate_node(_healthy(ecc_uncorrectable=1, disk_used_frac=0.9))
    primary = primary_diagnosis(h)
    assert primary.risk == Risk.HIGH


def test_pass_node_has_no_diagnosis():
    assert diagnose(evaluate_node(_healthy())) == []
    assert primary_diagnosis(evaluate_node(_healthy())) is None


def test_mock_provider_is_deterministic_and_varied():
    a = sweep(MockProvider(size=6))
    b = sweep(MockProvider(size=6))
    assert [n.to_dict() for n in a] == [n.to_dict() for n in b]  # reproducible
    verdicts = {evaluate_node(n).verdict for n in a}
    assert Verdict.PASS in verdicts and Verdict.FAIL in verdicts  # both outcomes present
