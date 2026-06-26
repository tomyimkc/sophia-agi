# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pluggable fleet telemetry providers (R1 read path).

A ``FleetProvider`` yields ``NodeMetrics`` for every node in the fleet. Two
implementations ship:

* ``MockProvider`` — a deterministic synthetic fleet (seeded, no network, no cost).
  This is the default so ``tools/cluster/inspect.py`` runs offline / airgapped, in
  the same spirit as the repo's ``--dry-run`` RunPod tools.
* ``RunPodProvider`` — maps the live RunPod REST ``GET /pods`` inventory (the same
  API and ``_api_request`` helper used by ``tools/runpod_rlvr.py``) onto
  ``NodeMetrics``.

**Honest bound:** the RunPod REST inventory exposes pod *status and shape*
(running/exited, GPU count, uptime), not on-die telemetry. Temperature, ECC, XID and
NVLink/RDMA counters require an on-node DCGM/`nvidia-smi` agent — so ``RunPodProvider``
leaves those fields ``None``, which the fail-closed evaluator correctly treats as
"unknown, can't clear", not "healthy". A future ``SSHProvider`` would fill them by
running ``nvidia-smi``/``dcgmi`` over the existing SSH lifecycle.
"""

from __future__ import annotations

import os
from typing import Iterable, Protocol

from agent.cluster.health import NodeMetrics


class FleetProvider(Protocol):
    """Anything that can enumerate node telemetry."""

    def list_nodes(self) -> list[NodeMetrics]:  # pragma: no cover - protocol
        ...


# A small deterministic synthetic fleet: a mix of healthy, warning and failing nodes
# so the inspection sweep and the exporter have realistic, reproducible signal. Index
# drives variation (no RNG → byte-identical across runs, like the repo's hash embedder).
def _synthetic_node(index: int) -> NodeMetrics:
    base = index % 6
    common = dict(node_id=f"gpu-node-{index:03d}", gpu_model="NVIDIA H100 80GB HBM3",
                  collected_at="1970-01-01T00:00:00+00:00")
    if base == 0:  # healthy
        return NodeMetrics(**common, gpu_temp_c=61.0, gpu_util=0.82, mem_used_frac=0.55,
                           disk_used_frac=0.40, ecc_uncorrectable=0, nvlink_down=0,
                           rdma_link_down=0, throttled=False)
    if base == 1:  # warm + disk filling — exercises the LOW-risk auto-heal path (gc_disk)
        return NodeMetrics(**common, gpu_temp_c=78.0, gpu_util=0.97, mem_used_frac=0.85,
                           disk_used_frac=0.88, ecc_uncorrectable=0, nvlink_down=0,
                           rdma_link_down=0, throttled=False)
    if base == 2:  # GPU fell off the bus (fatal XID 79)
        return NodeMetrics(**common, gpu_temp_c=58.0, gpu_util=0.0, mem_used_frac=0.10,
                           disk_used_frac=0.45, ecc_uncorrectable=0, xid_errors=(79,),
                           nvlink_down=0, rdma_link_down=0, throttled=False)
    if base == 3:  # RDMA flap — node isolates from the training fabric
        return NodeMetrics(**common, gpu_temp_c=64.0, gpu_util=0.30, mem_used_frac=0.60,
                           disk_used_frac=0.55, ecc_uncorrectable=0, nvlink_down=0,
                           rdma_link_down=2, throttled=False)
    if base == 4:  # uncorrectable ECC storm
        return NodeMetrics(**common, gpu_temp_c=70.0, gpu_util=0.45, mem_used_frac=0.70,
                           disk_used_frac=0.60, ecc_uncorrectable=3, xid_errors=(48, 94),
                           nvlink_down=0, rdma_link_down=0, throttled=False)
    # base == 5: unreachable
    return NodeMetrics(node_id=f"gpu-node-{index:03d}", gpu_model="NVIDIA H100 80GB HBM3",
                       reachable=False, collected_at="1970-01-01T00:00:00+00:00")


class MockProvider:
    """Deterministic synthetic fleet for offline inspection, tests and demos."""

    def __init__(self, size: int = 6) -> None:
        self.size = max(1, size)

    def list_nodes(self) -> list[NodeMetrics]:
        return [_synthetic_node(i) for i in range(self.size)]


def _pod_to_metrics(pod: dict) -> NodeMetrics:
    """Map one RunPod REST pod object onto NodeMetrics (status/shape only)."""

    status = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
    runtime = pod.get("runtime") or {}
    # Reachable iff RunPod reports the pod running; everything deeper is unknown.
    reachable = status == "RUNNING"
    gpu_model = None
    machine = pod.get("machine") or {}
    if isinstance(machine, dict):
        gpu_model = machine.get("gpuTypeId") or pod.get("gpuTypeId")
    return NodeMetrics(
        node_id=str(pod.get("id") or pod.get("name") or "unknown"),
        gpu_model=gpu_model,
        reachable=reachable,
        # DCGM-level fields intentionally left None → fail-closed "unknown".
        gpu_temp_c=None,
        ecc_uncorrectable=None,
        nvlink_down=None,
        rdma_link_down=None,
        collected_at=runtime.get("uptimeInSeconds") and None or None,
    )


class RunPodProvider:
    """Live fleet inventory via RunPod REST ``GET /pods``.

    Reuses the exact request helper and API base from ``tools/runpod_rlvr.py`` so
    auth, error handling and the endpoint stay in one place.
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise RuntimeError(
                "RunPodProvider needs RUNPOD_API_KEY (export it, or pass api_key=...). "
                "Use MockProvider for offline/dry-run inspection."
            )
        self.api_key = key

    def list_nodes(self) -> list[NodeMetrics]:
        from tools.runpod_rlvr import _api_request  # lazy: avoid import cost offline

        pods = _api_request("GET", "/pods", self.api_key, timeout=60) or []
        if isinstance(pods, dict):  # some API shapes wrap the list
            pods = pods.get("pods") or pods.get("data") or []
        return [_pod_to_metrics(p) for p in pods if isinstance(p, dict)]


def get_provider(source: str = "mock", *, size: int = 6,
                 inventory: str | None = None, ssh_key: str | None = None) -> FleetProvider:
    """Factory used by the CLIs. ``mock`` is the safe, offline default.

    * ``mock``    — deterministic synthetic fleet (offline).
    * ``runpod``  — live RunPod ``GET /pods`` inventory (status/shape only).
    * ``ssh``     — live DCGM-level telemetry over SSH. Needs an inventory (or
      ``SOPHIA_CLUSTER_INVENTORY``) of nodes, or falls back to RunPod discovery; the
      SSH key comes from ``ssh_key`` / ``SOPHIA_CLUSTER_SSH_KEY``.
    """

    source = (source or "mock").lower()
    if source == "mock":
        return MockProvider(size=size)
    if source == "runpod":
        return RunPodProvider()
    if source == "ssh":
        from agent.cluster.ssh_provider import SSHProvider

        inv = inventory or os.environ.get("SOPHIA_CLUSTER_INVENTORY")
        if inv:
            return SSHProvider.from_inventory(inv, key_path=ssh_key)
        # No inventory → discover targets from RunPod, probe them over SSH.
        return SSHProvider.from_runpod(key_path=ssh_key)
    raise ValueError(f"unknown fleet source: {source!r} (expected 'mock', 'runpod' or 'ssh')")


def sweep(provider: FleetProvider) -> list[NodeMetrics]:
    """Collect telemetry from every node (thin wrapper for symmetry/testability)."""

    nodes: Iterable[NodeMetrics] = provider.list_nodes()
    return list(nodes)
