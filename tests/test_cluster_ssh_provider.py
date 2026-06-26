# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the live SSH telemetry parser (pure; no network/GPUs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster.health import Verdict, evaluate_node  # noqa: E402
from agent.cluster.ssh_provider import (  # noqa: E402
    SSHProvider,
    SSHTarget,
    parse_probe,
    runpod_ssh_endpoint,
)

# A healthy 2-GPU node: 0x1 throttle = GPU idle (benign, NOT a fault).
HEALTHY = """\
===NVIDIA_SMI===
NVIDIA H100 80GB HBM3, 58, 80, 0, 0x0000000000000001
NVIDIA H100 80GB HBM3, 61, 85, 0, 0x0000000000000001
===DISK===
42
===MEM===
2000000000 1400000000
===DMESG===
===NVLINK===
GPU 0: NVIDIA H100
Link 0: 26.562 GB/s
Link 1: 26.562 GB/s
===IB===
4: ACTIVE
4: ACTIVE
===END===
"""

# GPU fell off the bus: XID 79 in dmesg + an RDMA link down + a real thermal throttle.
FAULTY = """\
===NVIDIA_SMI===
NVIDIA H100 80GB HBM3, 89, 0, 2, 0x0000000000000080
===DISK===
55
===MEM===
2000000000 900000000
===DMESG===
[12345.6] NVRM: Xid (PCI:0000:01:00): 79, pid=1234, GPU has fallen off the bus.
===NVLINK===
GPU 0: NVIDIA H100
Link 0: <inactive>
===IB===
4: ACTIVE
1: DOWN
===END===
"""


def test_parse_healthy_node():
    m = parse_probe("node-a", HEALTHY)
    assert m.reachable
    assert m.gpu_model == "NVIDIA H100 80GB HBM3"
    assert m.gpu_temp_c == 61.0          # max across GPUs
    assert m.gpu_util == 0.85            # max util / 100
    assert m.ecc_uncorrectable == 0
    assert m.xid_errors == ()
    assert m.throttled is False          # 0x1 idle bit is not a fault
    assert m.rdma_link_down == 0
    assert m.nvlink_down == 0
    assert abs(m.disk_used_frac - 0.42) < 1e-9
    assert abs(m.mem_used_frac - 0.30) < 1e-9
    assert evaluate_node(m).verdict == Verdict.PASS


def test_parse_faulty_node():
    m = parse_probe("node-b", FAULTY)
    assert m.gpu_temp_c == 89.0
    assert m.ecc_uncorrectable == 2
    assert m.xid_errors == (79,)
    assert m.throttled is True           # 0x80 HW thermal slowdown
    assert m.nvlink_down == 1
    assert m.rdma_link_down == 1
    h = evaluate_node(m)
    assert h.verdict == Verdict.FAIL
    signals = {r.signal for r in h.reasons}
    assert {"xid_errors", "ecc_uncorrectable", "rdma_link_down"} <= signals


def test_empty_output_is_unreachable():
    m = parse_probe("node-c", "")
    assert not m.reachable
    assert evaluate_node(m).verdict == Verdict.FAIL


def test_missing_sections_are_fail_closed_unknown():
    # Only nvidia-smi present, ECC reported N/A; NVLink/IB absent → unknown, not green.
    raw = "===NVIDIA_SMI===\nNVIDIA H100, 60, 50, [N/A], 0x0\n===END===\n"
    m = parse_probe("node-d", raw)
    assert m.reachable
    assert m.ecc_uncorrectable is None   # "[N/A]" → unknown
    assert m.nvlink_down is None         # section absent → unknown
    assert m.rdma_link_down is None
    # fail-closed: unknown critical metrics degrade the verdict to at least WARN
    assert evaluate_node(m).verdict in (Verdict.WARN, Verdict.FAIL)


def test_runpod_ssh_endpoint_extraction():
    pod = {"id": "p1", "publicIp": "1.2.3.4", "portMappings": {"22": 40022}}
    assert runpod_ssh_endpoint(pod) == ("1.2.3.4", 40022)
    assert runpod_ssh_endpoint({"id": "p2", "portMappings": {}}) == (None, None)


def test_from_inventory_builds_targets(tmp_path):
    inv = tmp_path / "fleet.json"
    inv.write_text(json.dumps([
        {"node_id": "n1", "host": "10.0.0.1", "port": 22},
        {"host": "10.0.0.2"},  # node_id defaults to host, port to 22
    ]))
    prov = SSHProvider.from_inventory(inv, key_path=str(tmp_path / "fake_key"))
    assert [t.node_id for t in prov.targets] == ["n1", "10.0.0.2"]
    assert prov.targets[1].port == 22


def test_ssh_argv_reuses_runpod_base(tmp_path):
    from agent.cluster.ssh_provider import _ssh_argv
    key = tmp_path / "k"
    argv = _ssh_argv(SSHTarget("n", "5.6.7.8", 2222), key)
    assert argv[0] == "ssh"
    assert "StrictHostKeyChecking=no" in argv
    assert "root@5.6.7.8" in argv
    assert str(key) in argv


def test_provider_requires_key(monkeypatch):
    monkeypatch.delenv("SOPHIA_CLUSTER_SSH_KEY", raising=False)
    try:
        SSHProvider([SSHTarget("n", "h")], key_path=None)
        assert False, "expected RuntimeError without a key"
    except RuntimeError as exc:
        assert "SSH private key" in str(exc)
