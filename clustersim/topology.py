# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cluster topology model — heterogeneous devices, nodes, NVLink islands, racks.

A first-principles, pure-stdlib model of the physical resource an AI supercomputer
schedules over. It is deliberately small but captures the structure that actually
drives placement quality: which GPUs share a fast intra-node link (an NVLink/NVSwitch
"island"), which nodes share a rack (cheaper cross-node hop), and the fact that a
real cluster mixes device *classes* (H100 / A100 / domestic NPU) with different VRAM.

The scheduler (clustersim/scheduler.py) consumes this to make topology-aware placement
decisions; the simulator (clustersim/simulator.py) replays jobs against it.

    cl = homogeneous_cluster(nodes=4, gpus_per_node=8, vram_gb=80, klass="H100")
    cl.total_gpus          # 32
    cl.free_gpus()         # all free at t=0
    cl.island_of("n0-g3")  # ("n0", 0)  -> node n0, island 0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Device:
    """One accelerator (or CPU/NPU) in the cluster."""

    id: str
    node_id: str
    rack_id: str
    kind: str            # "gpu" | "cpu" | "npu"
    klass: str           # microarch label, e.g. "H100", "A100", "domestic-x1"
    vram_gb: int
    island: int          # NVLink/NVSwitch island index *within* its node


@dataclass
class Node:
    """A physical server: a set of devices, an interconnect class, a rack."""

    id: str
    rack_id: str
    devices: list[Device]
    nic_gbps: float = 200.0          # per-node network egress (RoCEv2/IB class link)
    intra_link: str = "nvlink"       # intra-node fabric: "nvlink" | "pcie"

    def gpu_ids(self) -> list[str]:
        return [d.id for d in self.devices if d.kind == "gpu"]


@dataclass
class Cluster:
    """The whole machine. Owns device allocation state (free / busy)."""

    nodes: list[Node]
    _busy: set[str] = field(default_factory=set)

    # ----- static structure -----------------------------------------------
    def all_devices(self) -> list[Device]:
        return [d for n in self.nodes for d in n.devices]

    def gpus(self) -> list[Device]:
        return [d for d in self.all_devices() if d.kind == "gpu"]

    @property
    def total_gpus(self) -> int:
        return len(self.gpus())

    def device(self, dev_id: str) -> Device:
        for d in self.all_devices():
            if d.id == dev_id:
                return d
        raise KeyError(dev_id)

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def island_of(self, dev_id: str) -> tuple[str, int]:
        d = self.device(dev_id)
        return (d.node_id, d.island)

    # ----- allocation state -------------------------------------------------
    def free_gpus(self) -> list[Device]:
        return [d for d in self.gpus() if d.id not in self._busy]

    def busy_gpus(self) -> list[Device]:
        return [d for d in self.gpus() if d.id in self._busy]

    @property
    def free_count(self) -> int:
        return len(self.free_gpus())

    @property
    def utilization(self) -> float:
        """Fraction of GPUs currently allocated (0..1)."""
        return len(self._busy) / self.total_gpus if self.total_gpus else 0.0

    def allocate(self, dev_ids: Iterable[str]) -> None:
        dev_ids = list(dev_ids)
        for d in dev_ids:
            if d in self._busy:
                raise ValueError(f"device {d} already busy")
        self._busy.update(dev_ids)

    def release(self, dev_ids: Iterable[str]) -> None:
        for d in dev_ids:
            self._busy.discard(d)

    def fail_node(self, node_id: str) -> list[str]:
        """Mark every GPU on a node unavailable; return the freed busy ids.

        Used by the fault model. The devices stay *out* of free_gpus() because the
        node is down; callers also drop the node from `nodes` if it is a hard loss.
        """
        node = self.node(node_id)
        freed = [d.id for d in node.devices if d.id in self._busy]
        self.release(node.gpu_ids())
        self.nodes = [n for n in self.nodes if n.id != node_id]
        return freed


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def homogeneous_cluster(
    *,
    nodes: int,
    gpus_per_node: int,
    vram_gb: int = 80,
    klass: str = "H100",
    islands_per_node: int = 1,
    gpus_per_rack: int = 0,
    nic_gbps: float = 200.0,
) -> Cluster:
    """Build a uniform cluster: `nodes` nodes, `gpus_per_node` GPUs each.

    `islands_per_node` splits each node's GPUs into NVLink islands (e.g. an 8-GPU
    node with 2 islands = two 4-GPU NVSwitch domains). `gpus_per_rack` groups nodes
    into racks (0 = one rack for the whole cluster).
    """
    built: list[Node] = []
    gpus_seen = 0
    for ni in range(nodes):
        node_id = f"n{ni}"
        if gpus_per_rack:
            rack_id = f"r{(gpus_seen // gpus_per_rack)}"
        else:
            rack_id = "r0"
        devs: list[Device] = []
        for gi in range(gpus_per_node):
            island = gi // max(1, gpus_per_node // islands_per_node)
            devs.append(
                Device(
                    id=f"{node_id}-g{gi}",
                    node_id=node_id,
                    rack_id=rack_id,
                    kind="gpu",
                    klass=klass,
                    vram_gb=vram_gb,
                    island=min(island, islands_per_node - 1),
                )
            )
            gpus_seen += 1
        built.append(Node(id=node_id, rack_id=rack_id, devices=devs, nic_gbps=nic_gbps))
    return Cluster(nodes=built)


def heterogeneous_cluster(pools: list[dict]) -> Cluster:
    """Build a mixed-class cluster from a list of pool specs.

    Each pool: {"nodes": int, "gpus_per_node": int, "vram_gb": int, "klass": str,
                "islands_per_node": int (opt), "nic_gbps": float (opt)}.
    Pools are laid out in separate racks so the topology reflects the real thing:
    different accelerator generations rarely share a rack.

        heterogeneous_cluster([
            {"nodes": 2, "gpus_per_node": 8, "vram_gb": 80, "klass": "H100"},
            {"nodes": 2, "gpus_per_node": 8, "vram_gb": 64, "klass": "domestic-x1"},
        ])
    """
    built: list[Node] = []
    node_ix = 0
    for pi, pool in enumerate(pools):
        rack_id = f"r{pi}"
        for _ in range(pool["nodes"]):
            node_id = f"n{node_ix}"
            gpn = pool["gpus_per_node"]
            islands = pool.get("islands_per_node", 1)
            devs = [
                Device(
                    id=f"{node_id}-g{gi}",
                    node_id=node_id,
                    rack_id=rack_id,
                    kind="gpu",
                    klass=pool["klass"],
                    vram_gb=pool["vram_gb"],
                    island=min(gi // max(1, gpn // islands), islands - 1),
                )
                for gi in range(gpn)
            ]
            built.append(
                Node(
                    id=node_id,
                    rack_id=rack_id,
                    devices=devs,
                    nic_gbps=pool.get("nic_gbps", 200.0),
                )
            )
            node_ix += 1
    return Cluster(nodes=built)
