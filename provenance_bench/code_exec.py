# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Code-execution checking for the code-uplift benchmark — the strongest verifier.

Provenance reduces correctness to "does the documented author match". CODE reduces
it to something even harder and even more objective: *does the program pass its
tests when executed*. No judge, no lexical heuristic — the interpreter is ground
truth. This is the ideal verifiable signal (it is exactly how DeepSeek-R1 / RLVR
trains on code).

``run_solution(solution_code, test_code)`` writes ``solution + test`` to an
isolated temp dir and runs it; pass iff exit 0. ``extract_code`` pulls a fenced (or
raw) Python block from a model answer. Gated by ``SOPHIA_ALLOW_CODE_EXEC`` (default
on) and a wall-clock timeout; never touches the repo tree.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Reuse the agent verifier's extractor semantics, but keep a local copy so this
# module is import-light and usable from the benchmark without the full agent stack.
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


# A bare answer counts as code only if a LINE looks like a real Python statement —
# `def`/`class`/`import`/`from ... import`/`@decorator` at line start. The old
# heuristic matched the English words "return"/"import" anywhere, so ordinary prose
# ("In return, we...") was treated as code and fed to the executor. Anchored,
# multiline, statement-shaped only.
_BARE_CODE = re.compile(r"^\s*(?:def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\w+\s+import\s|@\w)", re.MULTILINE)


def extract_code(text: str) -> str:
    """Return the Python from a model answer: concatenated python-fenced blocks,
    else the raw text only if it is statement-shaped. Deterministic. Prose that
    merely contains the words 'return'/'import' is NOT treated as code."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return "\n\n".join(b.rstrip() for b in blocks)
    t = (text or "").strip()
    return t if _BARE_CODE.search(t) else ""


def _exec_on() -> bool:
    """Execution is OPT-IN: untrusted model code runs ONLY when SOPHIA_ALLOW_CODE_EXEC
    is explicitly truthy. Default OFF -> syntax-only (the security default; running
    arbitrary model output is not sandboxed, so it must never be the implicit path)."""
    return os.environ.get("SOPHIA_ALLOW_CODE_EXEC", "0").strip().lower() in ("1", "true", "yes", "on")


def run_solution(solution_code: str, test_code: str, *, timeout_sec: int = 15) -> dict:
    """Run ``solution_code`` + ``test_code`` together; pass iff exit 0.

    ``test_code`` is the HIDDEN canonical test (asserts calling the entry point) —
    it is appended after the model's solution so the model never sees it. Returns
    ``{passed, reason, executed}``. When execution is disabled (SOPHIA_ALLOW_CODE_EXEC=0)
    falls back to a syntax-only compile of the solution (can't verify correctness,
    so it is reported executed=False and passes only if it compiles).
    """
    if not solution_code.strip():
        return {"passed": False, "reason": "no code in answer", "executed": False}

    program = solution_code.rstrip() + "\n\n" + test_code.lstrip()
    if not _exec_on():
        try:
            compile(program, "<sol>", "exec")
            return {"passed": True, "reason": "syntax-only (exec disabled; set SOPHIA_ALLOW_CODE_EXEC=1)", "executed": False}
        except SyntaxError as exc:
            return {"passed": False, "reason": f"syntax error: {exc}", "executed": False}

    # SECURITY: this runs untrusted, model-generated code. It is NOT a real sandbox
    # (no container/seccomp); it is opt-in (_exec_on), time-boxed, and runs in its
    # own process GROUP so a timeout kills any grandchildren the code spawned. For
    # untrusted models, run the whole harness inside a container/VM.
    import signal

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prog.py"
        path.write_text(program, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(path)], cwd=tmp, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,  # own process group -> killpg on timeout
        )
        try:
            out, err = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            return {"passed": False, "reason": f"timed out after {timeout_sec}s", "executed": True}
    if proc.returncode == 0:
        return {"passed": True, "reason": "tests passed", "executed": True}
    tail = (err or out or "").strip().splitlines()
    return {"passed": False, "reason": (tail[-1] if tail else f"exit {proc.returncode}"), "executed": True}


def check_answer(answer: str, test_code: str, *, timeout_sec: int = 15) -> dict:
    """Extract code from a model ``answer`` and run it against ``test_code``."""
    return run_solution(extract_code(answer), test_code, timeout_sec=timeout_sec)
