# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hard-oracle code verification: sandboxed subprocess execution.

Runs model-produced code plus hidden tests in an isolated temp directory with
wall-clock timeout and POSIX rlimits (CPU time, address space). No network is
configured in the child environment. Execution is opt-in via
``SOPHIA_ALLOW_CODE_EXEC`` (default off) — when disabled, only syntax compile
is attempted and the verdict is ``abstain``.
"""
from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from provenance_bench.code_exec import extract_code

Verdict = Literal["accepted", "rejected", "abstain"]


def _exec_enabled() -> bool:
    return os.environ.get("SOPHIA_ALLOW_CODE_EXEC", "0").strip().lower() in ("1", "true", "yes", "on")


def _child_limits(cpu_sec: int, mem_mb: int) -> None:
    """Apply rlimits in the child before exec (best-effort; platform-dependent)."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 1))
    except (ValueError, OSError):
        pass
    if mem_mb > 0:
        try:
            cap = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ValueError, OSError):
            pass


def _blocked_env() -> dict[str, str]:
    """Minimal env: no proxy vars, no user site customization."""
    keep = {"PATH", "SYSTEMROOT", "HOME", "LANG", "LC_ALL", "PYTHONIOENCODING"}
    return {k: v for k, v in os.environ.items() if k in keep}


def run_program(
    solution_code: str,
    test_code: str,
    *,
    timeout_sec: int = 15,
    cpu_sec: int = 10,
    mem_mb: int = 256,
) -> dict[str, Any]:
    """Execute ``solution_code`` + hidden ``test_code``; return pass metadata."""
    if not solution_code.strip():
        return {"passed": False, "executed": False, "reason": "no code in answer"}
    program = solution_code.rstrip() + "\n\n" + test_code.lstrip()
    if not _exec_enabled():
        try:
            compile(program, "<sol>", "exec")
            return {
                "passed": False,
                "executed": False,
                "reason": "execution disabled (set SOPHIA_ALLOW_CODE_EXEC=1)",
            }
        except SyntaxError as exc:
            return {"passed": False, "executed": False, "reason": f"syntax error: {exc}"}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prog.py"
        path.write_text(program, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-I", str(path)],
            cwd=tmp,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=_blocked_env(),
            preexec_fn=lambda: _child_limits(cpu_sec, mem_mb),
        )
        try:
            out, err = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            return {"passed": False, "executed": True, "reason": f"timed out after {timeout_sec}s"}
    if proc.returncode == 0:
        return {"passed": True, "executed": True, "reason": "tests passed"}
    tail = (err or out or "").strip().splitlines()
    return {
        "passed": False,
        "executed": True,
        "reason": tail[-1] if tail else f"exit {proc.returncode}",
    }


def verify(
    answer: str,
    test_code: str,
    *,
    timeout_sec: int = 15,
    cpu_sec: int = 10,
    mem_mb: int = 256,
) -> dict[str, Any]:
    """Verify ``answer`` code against hidden ``test_code``.

    Returns ``{verdict, reasons, detail}``.
    """
    code = extract_code(answer or "")
    run = run_program(code, test_code, timeout_sec=timeout_sec, cpu_sec=cpu_sec, mem_mb=mem_mb)
    if not run.get("executed") and "disabled" in str(run.get("reason", "")):
        return {
            "verdict": "abstain",
            "reasons": [str(run["reason"])],
            "detail": {"executed": False},
        }
    if run.get("passed"):
        return {
            "verdict": "accepted",
            "reasons": [],
            "detail": {"executed": run.get("executed"), "reason": run.get("reason")},
        }
    return {
        "verdict": "rejected",
        "reasons": [str(run.get("reason", "tests failed"))],
        "detail": {"executed": run.get("executed")},
    }
