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


def extract_code(text: str) -> str:
    """Return the Python from a model answer: concatenated fenced blocks, else the
    raw text (a model may answer with bare code). Deterministic."""
    blocks = _FENCE.findall(text or "")
    if blocks:
        return "\n\n".join(b.rstrip() for b in blocks)
    # no fence — assume the whole answer is code if it looks like it
    t = (text or "").strip()
    return t if re.search(r"\bdef\s+\w+\s*\(|\bclass\s+\w+|return\b|import\b", t) else ""


def _exec_off() -> bool:
    return os.environ.get("SOPHIA_ALLOW_CODE_EXEC", "1").strip() in ("0", "false", "no")


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
    if _exec_off():
        try:
            compile(program, "<sol>", "exec")
            return {"passed": True, "reason": "syntax-only (exec disabled)", "executed": False}
        except SyntaxError as exc:
            return {"passed": False, "reason": f"syntax error: {exc}", "executed": False}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prog.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(path)], cwd=tmp, text=True,
                capture_output=True, timeout=timeout_sec, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "reason": f"timed out after {timeout_sec}s", "executed": True}
    if proc.returncode == 0:
        return {"passed": True, "reason": "tests passed", "executed": True}
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {"passed": False, "reason": (tail[-1] if tail else f"exit {proc.returncode}"), "executed": True}


def check_answer(answer: str, test_code: str, *, timeout_sec: int = 15) -> dict:
    """Extract code from a model ``answer`` and run it against ``test_code``."""
    return run_solution(extract_code(answer), test_code, timeout_sec=timeout_sec)
