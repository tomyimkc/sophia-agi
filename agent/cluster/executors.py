# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Real remediation executors (R4): drain / cordon / node-local fixes.

These are the *hands* the gated planner (``agent/cluster/heal.py``) calls once an
action is approved. Each executor is a ``Callable[[str, str], bool]`` — ``(node_id,
action) -> ok`` — so it plugs straight into ``plan_remediation(executor=...)`` and the
human-approved manual path. Three real backends ship plus a no-op:

* ``KubeExecutor``  — ``kubectl cordon`` / ``kubectl drain`` (scheduler drain/cordon).
* ``SlurmExecutor`` — ``scontrol update ... state=drain`` / ``scontrol reboot``.
* ``SSHExecutor``   — node-local fixes over SSH (gc disk, restart DCGM, restart fabric).
* ``NoopExecutor``  — logs the intended commands and reports success without running
  them (simulation for demos/tests).

Safety: the executor never decides *whether* to act — that is the gate's job
(``RemediationGate`` auto-approves only LOW-risk/auto-safe actions; drain/cordon always
ESCALATE to a human). The executor only carries out an already-approved action, and the
underlying command runner is injectable so every mapping is unit-tested without
touching a real cluster. An unsupported (node_id, action) pair returns ``False`` rather
than silently "succeeding".
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.cluster.ssh_provider import SSHTarget, _ssh_argv

# A runner executes one local command and returns (returncode, combined_output).
Runner = Callable[[list[str]], "tuple[int, str]"]


def default_runner(argv: list[str]) -> tuple[int, str]:
    """Run a local command, capturing output. Used by kube/slurm/ssh backends."""

    try:
        proc = subprocess.run(argv, text=True, capture_output=True, timeout=300)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 1, f"runner error: {exc}"
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


@dataclass
class _BaseExecutor:
    """Maps (node, action) → one or more commands; runs them; reports success."""

    runner: Runner = default_runner
    dry_run: bool = False

    name: str = "base"

    def _commands(self, node_id: str, action: str) -> list[list[str]] | None:
        raise NotImplementedError

    def __call__(self, node_id: str, action: str) -> bool:
        cmds = self._commands(node_id, action)
        if cmds is None:
            return False  # unsupported action for this backend — never a false success
        ok = True
        for cmd in cmds:
            if self.dry_run:
                print(f"[{self.name}:dry-run] {' '.join(shlex.quote(c) for c in cmd)}")
                continue
            rc, out = self.runner(cmd)
            if rc != 0:
                ok = False
                break
        return ok


class NoopExecutor(_BaseExecutor):
    """Logs intent, runs nothing, reports success (simulation backend)."""

    name = "noop"

    def _commands(self, node_id: str, action: str) -> list[list[str]] | None:
        print(f"[noop] would remediate {node_id} via action '{action}'")
        return []  # empty command list ⇒ __call__ returns True without running anything


@dataclass
class KubeExecutor(_BaseExecutor):
    """Drain / cordon a Kubernetes node. ``node_name`` maps node_id → k8s node name."""

    name: str = "kube"
    kubectl: str = "kubectl"
    node_name: Callable[[str], str] | None = None
    grace_period: int = 120

    def _k8s_node(self, node_id: str) -> str:
        return self.node_name(node_id) if self.node_name else node_id

    def _commands(self, node_id: str, action: str) -> list[list[str]] | None:
        node = self._k8s_node(node_id)
        if action == "cordon_and_investigate":
            return [[self.kubectl, "cordon", node]]
        if action in ("drain_and_reboot", "drain_and_diag"):
            return [[self.kubectl, "drain", node, "--ignore-daemonsets",
                     "--delete-emptydir-data", "--force",
                     f"--grace-period={self.grace_period}"]]
        if action == "observe":
            return [[self.kubectl, "describe", "node", node]]
        return None  # node-local actions (gc_disk, restart_*) are not the scheduler's job


@dataclass
class SlurmExecutor(_BaseExecutor):
    """Drain / reboot a Slurm node via scontrol."""

    name: str = "slurm"
    scontrol: str = "scontrol"

    def _commands(self, node_id: str, action: str) -> list[list[str]] | None:
        drain = [self.scontrol, "update", f"nodename={node_id}", "state=drain",
                 f"reason=sophia:{action}"]
        if action == "cordon_and_investigate":
            return [drain]
        if action == "drain_and_diag":
            return [drain]
        if action == "drain_and_reboot":
            return [drain, [self.scontrol, "reboot", node_id]]
        if action == "observe":
            return [[self.scontrol, "show", "node", node_id]]
        return None


@dataclass
class SSHExecutor(_BaseExecutor):
    """Run node-local remediation commands over SSH (reuses the SSH lifecycle).

    ``targets`` maps node_id → SSHTarget. Command templates are conservative and
    operator-overridable via ``templates``; the defaults avoid destructive deletes.
    """

    name: str = "ssh"
    targets: dict[str, SSHTarget] | None = None
    key_path: str | Path | None = None
    templates: dict[str, str] | None = None

    DEFAULT_TEMPLATES = {
        # Reclaim disk without touching user data: vacuum journald, prune docker, tmp.
        "gc_disk": ("journalctl --vacuum-time=2d 2>/dev/null; "
                    "docker system prune -f 2>/dev/null; "
                    "find /tmp -maxdepth 1 -name '*.tmp' -delete 2>/dev/null; true"),
        # Restart the DCGM host engine + persistence daemon so telemetry reports again.
        "restore_telemetry": ("nv-hostengine -t 2>/dev/null; nv-hostengine 2>/dev/null; "
                              "systemctl restart nvidia-dcgm 2>/dev/null; "
                              "nvidia-persistenced 2>/dev/null; true"),
        # Bounce the IB stack to recover a flapped RDMA link.
        "restart_fabric_iface": ("systemctl restart openibd 2>/dev/null || "
                                 "(modprobe -r ib_ipoib 2>/dev/null && modprobe ib_ipoib 2>/dev/null); true"),
        # Capture a full diagnostic snapshot for later root-cause.
        "observe": "nvidia-smi -q > /var/log/sophia_observe.log 2>&1; dmesg | tail -200 >> /var/log/sophia_observe.log; true",
    }

    def _resolve_key(self) -> Path:
        key = self.key_path or os.environ.get("SOPHIA_CLUSTER_SSH_KEY")
        if not key:
            raise RuntimeError("SSHExecutor needs key_path= or SOPHIA_CLUSTER_SSH_KEY")
        return Path(key)

    def _commands(self, node_id: str, action: str) -> list[list[str]] | None:
        templates = {**self.DEFAULT_TEMPLATES, **(self.templates or {})}
        remote = templates.get(action)
        if remote is None:
            return None  # drain/cordon are the scheduler's job, not a node-local fix
        target = (self.targets or {}).get(node_id)
        if target is None:
            target = SSHTarget(node_id=node_id, host=node_id)  # assume id == host
        argv = _ssh_argv(target, self._resolve_key()) + [remote]
        return [argv]


_BACKENDS = {
    "noop": NoopExecutor,
    "kube": KubeExecutor,
    "slurm": SlurmExecutor,
    "ssh": SSHExecutor,
}


def get_executor(backend: str = "noop", **cfg) -> _BaseExecutor:
    """Factory: ``get_executor('kube', node_name=...)`` etc. ``noop`` is the default."""

    backend = (backend or "noop").lower()
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise ValueError(f"unknown executor backend: {backend!r} "
                         f"(expected one of {sorted(_BACKENDS)})")
    return cls(**cfg)
