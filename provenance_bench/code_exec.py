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


# A runner template whose VERDICT is an unforgeable token, not the process exit
# code. The token is delivered on stdin (which the runner consumes BEFORE running
# the solution) and lives only in the runner's local frame — the solution, exec'd
# in a separate globals dict, can neither read it from a file (it is not in any
# file) nor from stdin (already consumed) nor from the runner's locals. It is
# written ONLY after every assertion in ``test_code`` passes. Therefore any
# premature process death — ``sys.exit``/``os._exit``/``atexit``/``raise
# SystemExit``, however it is triggered (directly, via ``exec``, or via a decoded
# payload) — prevents the token and reads as FAIL. This converts the dominant
# reward-hack class from "detected" to "structurally impossible".
# Optional return-hardening: after the solution is defined, every function it
# exposed is wrapped so its return value is forced through ``_canonical`` — exact
# plain scalars pass, containers are REBUILT as plain (stripping any subclass), and
# anything else (a custom object, e.g. one whose ``__eq__`` always returns True)
# raises, which fails the test. This makes the equality-override hack a STRUCTURAL
# failure at the grader, not merely a syntactic one the static scan must recognize.
_HARDEN_SRC = (
    "import types as _types\n"
    "def _canonical(v, _d=0):\n"
    "    if _d > 64: raise ValueError('return nested too deep')\n"
    "    if v is None or type(v) in (int, float, bool, str, bytes): return v\n"
    "    if isinstance(v, (list, tuple)):\n"
    "        _s = [_canonical(x, _d + 1) for x in v]\n"
    "        return tuple(_s) if isinstance(v, tuple) else _s\n"
    "    if isinstance(v, (set, frozenset)): return {_canonical(x, _d + 1) for x in v}\n"
    "    if isinstance(v, dict): return {_canonical(k, _d + 1): _canonical(w, _d + 1) for k, w in v.items()}\n"
    "    raise TypeError('non-plain return type: ' + type(v).__name__)\n"
    "def _wrap(_f):\n"
    "    def _w(*a, **k): return _canonical(_f(*a, **k))\n"
    "    return _w\n"
    "for _n, _v in list(_g.items()):\n"
    "    if isinstance(_v, _types.FunctionType): _g[_n] = _wrap(_v)\n"
)
_RUNNER_HEAD = (
    "import sys\n"
    "_tok = sys.stdin.readline().strip()\n"   # consume the secret first
    "_g = {'__name__': '__sol__'}\n"
    "try:\n"
    "    exec(compile(_SOLUTION, 'solution', 'exec'), _g)\n"
)
_RUNNER_TAIL = (
    "    exec(compile(_TEST, 'test', 'exec'), _g)\n"
    "except BaseException:\n"
    "    sys.exit(1)\n"                         # any error/exit -> no token -> FAIL
    "sys.stdout.write(_tok)\n"
    "sys.stdout.flush()\n"
)


def run_solution_isolated(solution_code: str, test_code: str, *, timeout_sec: int = 15,
                          harden_returns: bool = True) -> dict:
    """Grade ``solution_code`` against ``test_code`` with an exit-code-proof harness.

    Unlike :func:`run_solution` (whose pass/fail is the process exit status, and is
    therefore hackable by a solution that exits 0 before the test fails), the verdict
    here is whether a per-call secret token reaches stdout, which only happens after
    every assertion passes. Structurally defeats the whole process-death family.

    With ``harden_returns`` (default), every function the solution exposes is wrapped
    so its return is canonicalized to plain data before the test compares it — so an
    ``__eq__``-always-True object (or a list subclass with a rigged ``__eq__``) raises
    and fails, making equality-override a STRUCTURAL grader failure rather than only a
    syntactic one the static scan must catch. (Limit: a solution that passes custom
    objects *between its own helpers* and returns plain data is unaffected; one that
    returns a non-plain object fails — correct for a code task, whose answer is data.)

    Same ``SOPHIA_ALLOW_CODE_EXEC`` opt-in and syntax-only fallback contract as
    :func:`run_solution`. Still NOT a real sandbox — run untrusted models in a VM.
    """
    if not solution_code.strip():
        return {"passed": False, "reason": "no code in answer", "executed": False}

    if not _exec_on():
        program = solution_code.rstrip() + "\n\n" + test_code.lstrip()
        try:
            compile(program, "<sol>", "exec")
            return {"passed": True, "reason": "syntax-only (exec disabled; set SOPHIA_ALLOW_CODE_EXEC=1)", "executed": False}
        except SyntaxError as exc:
            return {"passed": False, "reason": f"syntax error: {exc}", "executed": False}

    import signal

    token = "OK_" + os.urandom(16).hex()
    harden = "".join("    " + ln + "\n" for ln in _HARDEN_SRC.splitlines()) if harden_returns else ""
    runner = (
        "_SOLUTION = " + repr(solution_code) + "\n"
        "_TEST = " + repr(test_code) + "\n"
        + _RUNNER_HEAD + harden + _RUNNER_TAIL
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runner.py"
        path.write_text(runner, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(path)], cwd=tmp, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(input=token + "\n", timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            return {"passed": False, "reason": f"timed out after {timeout_sec}s", "executed": True}
    if token in (out or ""):
        return {"passed": True, "reason": "tests passed (isolated)", "executed": True}
    tail = (err or "").strip().splitlines()
    return {"passed": False, "reason": (tail[-1] if tail else "no pass-token (failed/exited before tests)"), "executed": True}


def check_answer_isolated(answer: str, test_code: str, *, timeout_sec: int = 15) -> dict:
    """Extract code from ``answer`` and grade it with the exit-code-proof harness."""
    return run_solution_isolated(extract_code(answer), test_code, timeout_sec=timeout_sec)
