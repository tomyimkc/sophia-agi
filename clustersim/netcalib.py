# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Network-tax calibration — turn the simulator's modeled comm penalty into a measured one.

The simulator (clustersim/simulator.py) inflates a collective-heavy job's runtime when it is
spread across NVLink islands / nodes: effective = nominal * (1 + island_tax*(islands-1) +
node_tax*(nodes-1)). Until now island_tax/node_tax were guessed constants. This module
*derives* them from all-reduce cost at each interconnect tier, so the coefficients are
grounded in bandwidth instead of vibes.

Physical basis — ring all-reduce moves 2(N-1)/N · size bytes per rank over the bottleneck
link, plus 2(N-1) latency hops:

    T(size, N, bw, lat) = 2(N-1)/N · size/bw  +  2(N-1)·lat

A training step is `compute + comm`; `comm_fraction` is the comm share at full NVLink speed.
Spreading the job drops the bottleneck bandwidth (NVLink → NVSwitch/PCIe → NIC), inflating
only the comm part, so the per-hop runtime tax is:

    island_tax = comm_fraction · (T_cross_island / T_nvlink − 1)
    node_tax   = comm_fraction · (T_nic         / T_nvlink − 1)

Provenance is tracked per tier: a real NCCL benchmark (tools/bench_nccl_allreduce.py) can
replace the intra-node NVLink bandwidth with a *measured* number while cross-node stays
*modeled* from documented RoCEv2/IB line rates — an honest, partial calibration. Pure stdlib.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB_PATH = ROOT / "clustersim" / "netcalib.json"

# Documented, illustrative interconnect bandwidths (GB/s) and latencies (microseconds).
# These are the MODELED defaults; `source` records whether a tier was measured instead.
# nvlink   — intra-island GPU↔GPU effective all-reduce bus bandwidth (NVLink/NVSwitch domain)
# nvswitch — cross-island within one node (still on-node fabric, lower than a single island)
# nic      — cross-node over RoCEv2/InfiniBand (200 Gbps ≈ 25 GB/s line rate class)
DEFAULT_TIERS: dict[str, dict] = {
    "nvlink":   {"bandwidth_gbps": 400.0, "latency_us": 2.0,  "source": "modeled"},
    "nvswitch": {"bandwidth_gbps": 120.0, "latency_us": 4.0,  "source": "modeled"},
    "nic":      {"bandwidth_gbps": 50.0,  "latency_us": 8.0,  "source": "modeled"},  # 400 Gbps RoCE class
}

# Representative all-reduce operating point used to fit the coefficients.
DEFAULT_SIZE_BYTES = 256 * 1024 * 1024   # 256 MB gradient bucket
DEFAULT_N_RANKS = 8
# Exposed (non-overlapped) comm share of a step at full NVLink speed. This is a worst-case
# comm-bound knob: real training hides much of it via compute/comm overlap and by keeping
# bandwidth-heavy collectives intra-node, so treat the derived node_tax as an upper bound.
DEFAULT_COMM_FRACTION = 0.15


@dataclass
class LinkTier:
    name: str
    bandwidth_gbps: float
    latency_us: float
    source: str = "modeled"      # "measured" | "modeled"


def ring_allreduce_time_s(size_bytes: int, n_ranks: int, bandwidth_gbps: float,
                          latency_us: float = 0.0) -> float:
    """Ring all-reduce wall time (s). 1 rank → 0 (nothing to reduce)."""
    if n_ranks <= 1:
        return 0.0
    bw_bytes_per_s = bandwidth_gbps * 1e9
    transfer = (2.0 * (n_ranks - 1) / n_ranks) * size_bytes / bw_bytes_per_s
    latency = 2.0 * (n_ranks - 1) * latency_us * 1e-6
    return transfer + latency


def bus_bandwidth_gbps(size_bytes: int, n_ranks: int, time_s: float) -> float:
    """Invert a measured all-reduce time into NCCL 'bus bandwidth' (GB/s).

    busbw = algbw · 2(N-1)/N, where algbw = size / time. This is the standard
    nccl-tests convention, so measured numbers are directly comparable.
    """
    if time_s <= 0 or n_ranks <= 1:
        return 0.0
    algbw = size_bytes / time_s
    return (algbw * 2.0 * (n_ranks - 1) / n_ranks) / 1e9


@dataclass
class Calibration:
    tiers: dict[str, LinkTier]
    size_bytes: int
    n_ranks: int
    comm_fraction: float
    island_tax: float
    node_tax: float
    source: str                     # overall: "measured" if any tier measured, else "modeled"
    notes: str = ""
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema": "sophia.cluster_netcalib.v1",
            "source": self.source,
            "size_bytes": self.size_bytes,
            "n_ranks": self.n_ranks,
            "comm_fraction": self.comm_fraction,
            "island_tax": round(self.island_tax, 6),
            "node_tax": round(self.node_tax, 6),
            "tiers": {k: asdict(v) for k, v in self.tiers.items()},
            "notes": self.notes,
            "provenance": self.provenance,
        }

    def save(self, path: str | Path = DEFAULT_CALIB_PATH) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return p


def fit_tax(tiers: dict[str, LinkTier], *, size_bytes: int = DEFAULT_SIZE_BYTES,
            n_ranks: int = DEFAULT_N_RANKS,
            comm_fraction: float = DEFAULT_COMM_FRACTION,
            notes: str = "") -> Calibration:
    """Derive island_tax / node_tax from per-tier all-reduce cost (see module docstring)."""
    t = {name: ring_allreduce_time_s(size_bytes, n_ranks, tier.bandwidth_gbps, tier.latency_us)
         for name, tier in tiers.items()}
    t_island = t["nvlink"]
    if t_island <= 0:
        raise ValueError("nvlink tier all-reduce time is zero; check bandwidth/ranks")
    island_tax = comm_fraction * (t["nvswitch"] / t_island - 1.0)
    node_tax = comm_fraction * (t["nic"] / t_island - 1.0)
    overall = "measured" if any(x.source == "measured" for x in tiers.values()) else "modeled"
    return Calibration(
        tiers=tiers,
        size_bytes=size_bytes,
        n_ranks=n_ranks,
        comm_fraction=comm_fraction,
        island_tax=max(0.0, island_tax),
        node_tax=max(0.0, node_tax),
        source=overall,
        notes=notes,
        provenance={k: v.source for k, v in tiers.items()},
    )


def default_modeled(**kw) -> Calibration:
    """The committed, reproducible calibration from documented (modeled) bandwidths."""
    tiers = {name: LinkTier(name=name, **spec) for name, spec in DEFAULT_TIERS.items()}
    return fit_tax(
        tiers,
        notes=("MODELED from documented interconnect line rates — not a measurement. "
               "Run tools/runpod_nccl_bench.py to replace the nvlink tier with measured data."),
        **kw,
    )


def from_nccl_report(report: dict, *, comm_fraction: float = DEFAULT_COMM_FRACTION,
                     pick_size_bytes: int | None = None) -> Calibration:
    """Build a calibration from a tools/bench_nccl_allreduce.py result report.

    The benchmark measures intra-node NVLink bus bandwidth at one or more message sizes;
    that *measured* bandwidth replaces the nvlink tier. Cross-island/cross-node tiers stay
    MODELED (a single pod cannot measure them), and provenance records the mix honestly.
    """
    rows = report.get("results", [])
    if not rows:
        raise ValueError("nccl report has no results")
    size = pick_size_bytes or max(r["size_bytes"] for r in rows)
    row = min(rows, key=lambda r: abs(r["size_bytes"] - size))
    measured_bw = float(row.get("busbw_gbps") or
                        bus_bandwidth_gbps(row["size_bytes"], report["n_ranks"], row["time_s"]))
    tiers = {name: LinkTier(name=name, **spec) for name, spec in DEFAULT_TIERS.items()}
    tiers["nvlink"] = LinkTier(name="nvlink", bandwidth_gbps=measured_bw,
                               latency_us=DEFAULT_TIERS["nvlink"]["latency_us"], source="measured")
    return fit_tax(
        tiers,
        size_bytes=row["size_bytes"],
        n_ranks=report["n_ranks"],
        comm_fraction=comm_fraction,
        notes=(f"nvlink tier MEASURED on {report.get('gpu_name', 'unknown GPU')} "
               f"({measured_bw:.1f} GB/s busbw, {report['n_ranks']} ranks); "
               "cross-island/cross-node tiers remain MODELED."),
    )


def load_calibration(path: str | Path = DEFAULT_CALIB_PATH) -> Calibration | None:
    """Load a saved calibration JSON, or None if absent/unreadable."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    tiers = {k: LinkTier(name=k, bandwidth_gbps=v["bandwidth_gbps"],
                         latency_us=v["latency_us"], source=v.get("source", "modeled"))
             for k, v in d.get("tiers", {}).items()}
    return Calibration(
        tiers=tiers,
        size_bytes=d["size_bytes"],
        n_ranks=d["n_ranks"],
        comm_fraction=d["comm_fraction"],
        island_tax=d["island_tax"],
        node_tax=d["node_tax"],
        source=d.get("source", "modeled"),
        notes=d.get("notes", ""),
        provenance=d.get("provenance", {}),
    )
