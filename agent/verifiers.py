"""Pluggable verifiers for the agent harness, eval, and RL loops.

A verifier is ``(text, task, step) -> {"passed": bool, "reasons": [...], "detail": {...}}``
matching agent.harness. These turn "looks good" into "verified": exact answers,
regex, keywords, unit tests, rubric scoring, and citation checks — plus
combinators. Quality follows verifiability.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from agent.config import ROOT

Verifier = Callable[[str, Any, dict], dict]


def _ok(detail: dict | None = None) -> dict:
    return {"passed": True, "reasons": [], "detail": detail or {}}


def _fail(reasons: list[str], detail: dict | None = None) -> dict:
    return {"passed": False, "reasons": reasons, "detail": detail or {}}


def exact_match(expected: str, *, normalize: bool = True) -> Verifier:
    """Pass if the answer equals (or, by default, contains) the expected string."""

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower() if normalize else s

    target = norm(expected)

    def _verify(text: str, task: Any, step: dict) -> dict:
        got = norm(text)
        if target == got or target in got:
            return _ok({"expected": expected})
        return _fail([f"expected '{expected}' not found"], {"expected": expected})

    return _verify


def regex_match(pattern: str, *, flags: int = re.IGNORECASE) -> Verifier:
    compiled = re.compile(pattern, flags)

    def _verify(text: str, task: Any, step: dict) -> dict:
        if compiled.search(text):
            return _ok({"pattern": pattern})
        return _fail([f"pattern /{pattern}/ did not match"], {"pattern": pattern})

    return _verify


def keyword(must_include: list[str] | None = None, must_avoid: list[str] | None = None) -> Verifier:
    must_include = must_include or []
    must_avoid = must_avoid or []

    def _verify(text: str, task: Any, step: dict) -> dict:
        lowered = text.lower()
        missing = [k for k in must_include if k.lower() not in lowered]
        present_forbidden = [k for k in must_avoid if k.lower() in lowered]
        reasons = [f"missing: {k}" for k in missing] + [f"forbidden present: {k}" for k in present_forbidden]
        return _ok() if not reasons else _fail(reasons, {"missing": missing, "forbidden": present_forbidden})

    return _verify


def unit_test(command: list[str], *, cwd: Path = ROOT, timeout_sec: int = 300) -> Verifier:
    """Pass iff a command (pytest/build/lint) exits 0. The answer text is ignored;
    the world state is the verifier (the strongest signal for code tasks)."""

    def _verify(text: str, task: Any, step: dict) -> dict:
        try:
            proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout_sec, check=False)
        except subprocess.TimeoutExpired:
            return _fail([f"command timed out: {' '.join(command)}"])
        except FileNotFoundError as exc:
            return _fail([f"command not found: {exc}"])
        if proc.returncode == 0:
            return _ok({"cmd": " ".join(command), "returncode": 0})
        return _fail([f"exit {proc.returncode}: {(proc.stderr or proc.stdout)[-300:]}"], {"returncode": proc.returncode})

    return _verify


def score_pack_case(case: dict, *, threshold: float = 1.0) -> Verifier:
    """Wrap hidden_eval_protocol.score_case as a verifier (rubric/operational)."""
    from tools.hidden_eval_protocol import score_case

    def _verify(text: str, task: Any, step: dict) -> dict:
        result = score_case(case, text)
        ratio = (result["score"] / result["maxPoints"]) if result.get("maxPoints") else 0.0
        passed = ratio >= threshold and result.get("passed", False)
        reasons = [m.get("label", str(m)) for m in result.get("missedRubric", [])]
        return {"passed": passed, "reasons": reasons, "detail": {"score": result["score"], "maxPoints": result["maxPoints"]}}

    return _verify


def citation_present(sources: list[str]) -> Verifier:
    """Pass if the answer cites at least one provided source label/path token."""
    tokens = [s for s in (Path(str(x)).name for x in sources) if s]

    def _verify(text: str, task: Any, step: dict) -> dict:
        lowered = text.lower()
        hit = any(tok.lower() in lowered for tok in tokens) or bool(re.search(r"\[(?:local|web)\s*\d+\]|source", lowered))
        return _ok() if hit else _fail(["no citation to a provided source"], {"sources": tokens})

    return _verify


def all_of(*verifiers: Verifier) -> Verifier:
    def _verify(text: str, task: Any, step: dict) -> dict:
        reasons: list[str] = []
        detail: dict[str, Any] = {}
        passed = True
        for i, v in enumerate(verifiers):
            r = v(text, task, step)
            detail[f"v{i}"] = r.get("detail", {})
            if not r["passed"]:
                passed = False
                reasons.extend(r["reasons"])
        return {"passed": passed, "reasons": reasons, "detail": detail}

    return _verify


def any_of(*verifiers: Verifier) -> Verifier:
    def _verify(text: str, task: Any, step: dict) -> dict:
        all_reasons: list[str] = []
        for v in verifiers:
            r = v(text, task, step)
            if r["passed"]:
                return _ok(r.get("detail", {}))
            all_reasons.extend(r["reasons"])
        return _fail(["none of the alternatives passed", *all_reasons])

    return _verify
