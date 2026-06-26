# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Live GPU telemetry over SSH (R1 read path, real fleet).

``SSHProvider`` fills the DCGM-level fields the ``RunPodProvider`` leaves ``None`` by
running a single probe script per node over SSH and parsing real ``nvidia-smi`` /
``dmesg`` / InfiniBand output into ``NodeMetrics``. It reuses the exact SSH lifecycle
the repo already trusts for training pods (``tools/runpod_rlvr.py``: ``PodConnection``,
``_ssh_base`` for the connection options, ``_api_request`` + pod port-mappings for
RunPod discovery).

Design:

* **One round trip per node.** The probe is a single bash script emitting sectioned
  ``===MARKER===`` output; ``parse_probe`` (a pure function) turns that text into
  ``NodeMetrics``. So all parsing is unit-testable offline with canned output — no
  network, no GPUs.
* **Fail-closed.** An SSH failure / timeout yields ``reachable=False`` (→ FAIL). A
  section that is missing or unparseable leaves its metric ``None`` (→ WARN, "unknown,
  can't clear"), never a false green.
* **Aggregation across GPUs** is worst-case: max temperature, max utilisation, summed
  uncorrectable-ECC, any-throttled — so one bad GPU degrades the node verdict.

Targets come from an explicit inventory (``from_inventory``) or live RunPod discovery
(``from_runpod``). The SSH private key is supplied via ``key_path`` or the
``SOPHIA_CLUSTER_SSH_KEY`` env var.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.cluster.health import NodeMetrics

# Throttle bitmask bits that indicate a *fault* (thermal/power slowdown), not benign
# idle. SW thermal 0x20 | HW slowdown 0x40 | HW thermal 0x80 | HW power brake 0x100.
_THROTTLE_FAULT_MASK = 0x20 | 0x40 | 0x80 | 0x100

# nvidia-smi query — keep names comma-free so CSV parsing stays trivial.
_SMI_QUERY = ("name,temperature.gpu,utilization.gpu,"
              "ecc.errors.uncorrected.aggregate.total,clocks_throttle_reasons.active")

# Single-round-trip remote probe. Tolerant: every command swallows its own failure so
# one missing tool never aborts the rest (the parser treats absent sections as unknown).
PROBE_SCRIPT = r"""
set +e
echo '===NVIDIA_SMI==='
nvidia-smi --query-gpu=NAME_TEMP_UTIL_ECC_THROTTLE --format=csv,noheader,nounits 2>/dev/null
echo '===DISK==='
df --output=pcent / 2>/dev/null | tail -1 | tr -dc '0-9'; echo
echo '===MEM==='
free -b 2>/dev/null | awk '/Mem:/ {print $2" "$7}'
echo '===DMESG==='
{ dmesg 2>/dev/null || journalctl -k --no-pager 2>/dev/null; } | grep 'NVRM: Xid' | tail -100
echo '===NVLINK==='
nvidia-smi nvlink --status 2>/dev/null
echo '===IB==='
cat /sys/class/infiniband/*/ports/*/state 2>/dev/null
echo '===END==='
""".replace("NAME_TEMP_UTIL_ECC_THROTTLE", _SMI_QUERY)

# Optional deep-diagnostic suffix: dcgmi diag at the requested run level, JSON output.
# Appended to PROBE_SCRIPT only when deep=True (the run is slow — minutes at -r 3).
_DCGM_SUFFIX = """
echo '===DCGM==='
dcgmi diag -r {level} -j 2>/dev/null
echo '===DCGM_END==='
"""


def _build_probe(deep: bool, dcgm_level: int) -> str:
    if not deep:
        return PROBE_SCRIPT
    # Insert the DCGM section just before the END marker.
    body = PROBE_SCRIPT.replace("echo '===END==='", "").rstrip()
    return body + "\n" + _DCGM_SUFFIX.format(level=dcgm_level) + "echo '===END==='\n"


def parse_dcgm_diag(text: str) -> tuple[str, ...] | None:
    """Parse ``dcgmi diag -j`` JSON → tuple of FAILED test names.

    Returns ``None`` if no parseable diagnostic was produced (not run / unavailable),
    ``()`` if it ran and every test passed, else the failing test names. Tolerant of
    the schema variations across DCGM versions (test_categories → tests → results).
    """

    text = (text or "").strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None

    # Unwrap the common top-level container key(s).
    root = data
    if isinstance(data, dict):
        for key in ("DCGM GPU Diagnostic", "DCGM Diagnostic", "diagnostic"):
            if key in data:
                root = data[key]
                break

    categories = []
    if isinstance(root, dict):
        categories = root.get("test_categories") or root.get("categories") or []
    if not isinstance(categories, list):
        return None

    failed: list[str] = []
    saw_any = False
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        for test in cat.get("tests", []) or []:
            if not isinstance(test, dict):
                continue
            name = test.get("name") or test.get("test_name") or "unknown"
            results = test.get("results")
            statuses = []
            if isinstance(results, list):
                statuses = [str(r.get("status", "")) for r in results if isinstance(r, dict)]
            elif "status" in test:
                statuses = [str(test["status"])]
            for st in statuses:
                saw_any = True
                if st.strip().lower() in ("fail", "failed", "error"):
                    failed.append(str(name))
                    break
    if not saw_any:
        return None
    # De-dup while preserving order.
    return tuple(dict.fromkeys(failed))


@dataclass(frozen=True)
class SSHTarget:
    node_id: str
    host: str
    port: int = 22
    user: str = "root"


def _split_sections(raw: str) -> dict[str, str]:
    """Split sectioned probe output into {marker: body}."""

    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in raw.splitlines():
        m = re.match(r"^===([A-Z_]+)===$", line.strip())
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None and current not in sections:
        sections[current] = "\n".join(buf).strip()
    return sections


def _to_int(text: str) -> int | None:
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return None


def _parse_throttle(token: str) -> bool | None:
    token = token.strip()
    try:
        bits = int(token, 16) if token.lower().startswith("0x") else int(token)
    except ValueError:
        return None
    return bool(bits & _THROTTLE_FAULT_MASK)


def parse_probe(node_id: str, raw: str, *, gpu_model: str | None = None,
                collected_at: str | None = None) -> NodeMetrics:
    """Pure parser: probe text → NodeMetrics. Fully testable without SSH."""

    if not raw or "===NVIDIA_SMI===" not in raw:
        # No usable telemetry came back → treat as unreachable (fail-closed FAIL).
        return NodeMetrics(node_id=node_id, gpu_model=gpu_model, reachable=False,
                           collected_at=collected_at)

    sec = _split_sections(raw)

    # --- nvidia-smi per-GPU rows: name, temp, util, ecc_uncorr, throttle_hex ---
    temps: list[float] = []
    utils: list[float] = []
    eccs: list[int] = []
    throttled_any = False
    throttle_known = False
    name: str | None = gpu_model
    for line in (sec.get("NVIDIA_SMI") or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gname, t, u, ecc, thr = parts[0], parts[1], parts[2], parts[3], parts[4]
        if name is None and gname and gname.upper() not in ("N/A", "[N/A]"):
            name = gname
        if (tv := _safe_float(t)) is not None:
            temps.append(tv)
        if (uv := _safe_float(u)) is not None:
            utils.append(uv)
        if (ev := _to_int(ecc)) is not None:
            eccs.append(ev)
        tr = _parse_throttle(thr)
        if tr is not None:
            throttle_known = True
            throttled_any = throttled_any or tr

    # --- disk percent → fraction ---
    disk_used_frac = None
    if (dp := _to_int(sec.get("DISK", ""))) is not None:
        disk_used_frac = dp / 100.0

    # --- host memory: "total available" (bytes) → used fraction ---
    mem_used_frac = None
    mem_line = (sec.get("MEM") or "").split()
    if len(mem_line) == 2:
        total, avail = _to_int(mem_line[0]), _to_int(mem_line[1])
        if total and avail is not None and total > 0:
            mem_used_frac = max(0.0, (total - avail) / total)

    # --- XID codes from dmesg NVRM lines ---
    xids = tuple(int(x) for x in re.findall(r"Xid\s*\([^)]*\):\s*(\d+)", sec.get("DMESG", "")))

    # --- NVLink: count inactive links (None if the command produced nothing) ---
    nvlink_body = sec.get("NVLINK", "")
    nvlink_down = None
    if nvlink_body:
        nvlink_down = len(re.findall(r"inactive", nvlink_body, re.IGNORECASE))

    # --- InfiniBand/RDMA: lines like "4: ACTIVE"; count non-ACTIVE (None if no IB) ---
    ib_body = sec.get("IB", "")
    rdma_link_down = None
    if ib_body:
        states = [ln.split(":")[-1].strip().upper() for ln in ib_body.splitlines() if ":" in ln]
        if states:
            rdma_link_down = sum(1 for s in states if s != "ACTIVE")

    # --- DCGM deep diagnostic (only present when probed with deep=True) ---
    dcgm_diag = parse_dcgm_diag(sec["DCGM"]) if "DCGM" in sec else None

    return NodeMetrics(
        node_id=node_id,
        gpu_model=name,
        reachable=True,
        gpu_temp_c=max(temps) if temps else None,
        gpu_util=(max(utils) / 100.0) if utils else None,
        mem_used_frac=mem_used_frac,
        disk_used_frac=disk_used_frac,
        # Summed across GPUs; empty (e.g. ECC reported "N/A") stays None → fail-closed WARN.
        ecc_uncorrectable=sum(eccs) if eccs else None,
        xid_errors=xids,
        throttled=throttled_any if throttle_known else None,
        nvlink_down=nvlink_down,
        rdma_link_down=rdma_link_down,
        dcgm_diag=dcgm_diag,
        collected_at=collected_at,
    )


def _safe_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


def _ssh_argv(target: SSHTarget, key_path: Path) -> list[str]:
    """SSH argv mirroring tools/runpod_rlvr._ssh_base (reused for the root case)."""

    from tools.runpod_rlvr import PodConnection, _ssh_base

    conn = PodConnection(pod_id=target.node_id, public_ip=target.host, ssh_port=target.port)
    if target.user == "root":
        return _ssh_base(conn, key_path)
    # Non-root: same options, different login (kept in sync with _ssh_base by intent).
    base = _ssh_base(conn, key_path)
    base[-1] = f"{target.user}@{target.host}"
    return base


class SSHProvider:
    """Collect live ``NodeMetrics`` from a fleet over SSH."""

    def __init__(self, targets: list[SSHTarget], *, key_path: str | Path | None = None,
                 timeout_s: int = 25, max_workers: int = 8,
                 deep: bool = False, dcgm_level: int = 1) -> None:
        if not targets:
            raise ValueError("SSHProvider needs at least one target")
        key = key_path or os.environ.get("SOPHIA_CLUSTER_SSH_KEY")
        if not key:
            raise RuntimeError(
                "SSHProvider needs an SSH private key (key_path= or SOPHIA_CLUSTER_SSH_KEY)."
            )
        self.targets = targets
        self.key_path = Path(key)
        # Deep dcgmi diag runs for minutes — give it a much longer timeout.
        self.timeout_s = timeout_s if not deep else max(timeout_s, 600)
        self.max_workers = max(1, min(max_workers, len(targets)))
        self.probe_script = _build_probe(deep, dcgm_level)

    def _probe_one(self, target: SSHTarget) -> NodeMetrics:
        argv = _ssh_argv(target, self.key_path) + [self.probe_script]
        try:
            proc = subprocess.run(argv, text=True, capture_output=True, timeout=self.timeout_s)
        except (subprocess.TimeoutExpired, OSError):
            return NodeMetrics(node_id=target.node_id, reachable=False)
        if proc.returncode != 0 and "===NVIDIA_SMI===" not in (proc.stdout or ""):
            return NodeMetrics(node_id=target.node_id, reachable=False)
        return parse_probe(target.node_id, proc.stdout or "")

    def list_nodes(self) -> list[NodeMetrics]:
        # Parallel sweep — fleet wall-clock is the slowest single node, not the sum.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            by_index = list(pool.map(self._probe_one, self.targets))
        return by_index

    # --- target discovery ----------------------------------------------------
    @classmethod
    def from_inventory(cls, path: str | Path, **kw) -> "SSHProvider":
        """Build from a JSON inventory: ``[{"node_id","host","port","user"}, ...]``."""

        import json

        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        targets = [
            SSHTarget(node_id=str(r.get("node_id") or r["host"]), host=r["host"],
                      port=int(r.get("port", 22)), user=str(r.get("user", "root")))
            for r in rows
        ]
        return cls(targets, **kw)

    @classmethod
    def from_runpod(cls, api_key: str | None = None, **kw) -> "SSHProvider":
        """Discover running RunPod pods and resolve their public SSH host/port."""

        from tools.runpod_rlvr import _api_request

        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise RuntimeError("from_runpod needs RUNPOD_API_KEY")
        pods = _api_request("GET", "/pods", key, timeout=60) or []
        if isinstance(pods, dict):
            pods = pods.get("pods") or pods.get("data") or []
        targets = []
        for pod in pods:
            if not isinstance(pod, dict):
                continue
            host, port = runpod_ssh_endpoint(pod)
            if host and port:
                targets.append(SSHTarget(node_id=str(pod.get("id") or pod.get("name")),
                                         host=host, port=port))
        if not targets:
            raise RuntimeError("no RunPod pods exposed a public SSH mapping")
        return cls(targets, **kw)


def runpod_ssh_endpoint(pod: dict[str, Any]) -> tuple[str | None, int | None]:
    """Extract (public_ip, ssh_port) from a RunPod pod object (pure; mirrors _poll_ssh)."""

    public_ip = pod.get("publicIp")
    pm = pod.get("portMappings") or {}
    ssh_port = pm.get("22") or pm.get(22) or pm.get("22/tcp") or pm.get("tcp/22")
    if public_ip and ssh_port:
        try:
            return str(public_ip), int(ssh_port)
        except (ValueError, TypeError):
            return None, None
    return None, None
