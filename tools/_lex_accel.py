#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Optional accelerator bridge to the `sophia-lex` Rust crate (tools/sophia-lex).

This is a THIN, FAIL-SAFE bridge. The pure-Python measurement-gate tools
(tools/lint_claims.py, tools/assert_decontam.py) remain the reference oracle and
the CI default. When the compiled `sophia-lex` binary is present, these helpers
shell out to it for the hot text-scan loops:

  * overclaim_scan  — the no-overclaim phrase scan (mirrors lint_claims FORBIDDEN)
  * decontam_near   — the content-shingle near-dup scan, with FULL eval coverage
                      (no `--max-eval-shingle` cap)

If the binary is missing or anything goes wrong, every helper raises
`LexUnavailable`; callers must catch it and fall back to Python. Nothing here is
allowed to make a gate pass that Python would fail — acceleration is opt-in
(`--accel`) and verified byte-for-byte against Python by tools/test_lex_parity.py.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "tools" / "sophia-lex"
DEFAULT_BIN = CRATE / "target" / "release" / "sophia-lex"


class LexUnavailable(RuntimeError):
    """Raised when the sophia-lex accelerator cannot be used; callers fall back."""


def binary_path() -> Path | None:
    """Locate the compiled binary: $SOPHIA_LEX_BIN, then the release build."""
    env = os.environ.get("SOPHIA_LEX_BIN")
    if env and Path(env).exists():
        return Path(env)
    if DEFAULT_BIN.exists():
        return DEFAULT_BIN
    return None


def available() -> bool:
    return binary_path() is not None


def build(release: bool = True) -> Path:
    """Build the crate (used by the parity test). Raises LexUnavailable if cargo
    is missing or the build fails — never installs a toolchain."""
    if shutil.which("cargo") is None:
        raise LexUnavailable("cargo not found; cannot build sophia-lex")
    cmd = ["cargo", "build"] + (["--release"] if release else [])
    proc = subprocess.run(cmd, cwd=CRATE, capture_output=True, text=True)
    if proc.returncode != 0:
        raise LexUnavailable(f"cargo build failed:\n{proc.stderr[-2000:]}")
    bin_ = binary_path()
    if bin_ is None:
        raise LexUnavailable("build succeeded but binary not found")
    return bin_


def _run(args: list[str], stdin: bytes | None = None) -> subprocess.CompletedProcess:
    bin_ = binary_path()
    if bin_ is None:
        raise LexUnavailable("sophia-lex binary not built (run cargo build --release in tools/sophia-lex)")
    try:
        return subprocess.run(
            [str(bin_), *args], input=stdin, capture_output=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:  # pragma: no cover - env dependent
        raise LexUnavailable(f"sophia-lex invocation failed: {exc}") from exc


def overclaim_scan(files: list[Path]) -> list[tuple[str, int, str]]:
    """Return (relpath, line, why) for every forbidden-phrase hit across `files`.
    Matches the triples tools/lint_claims.py reports for its FORBIDDEN scan."""
    proc = _run(["overclaim", *[str(f) for f in files]])
    if proc.returncode not in (0, 1):  # 0=clean, 1=violations; 2=usage error
        raise LexUnavailable(f"overclaim exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    out: list[tuple[str, int, str]] = []
    for raw in proc.stdout.decode("utf-8", "replace").splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        p = Path(obj["file"])
        try:
            rel = str(p.resolve().relative_to(ROOT))
        except ValueError:
            rel = obj["file"]  # outside the repo (e.g. a temp fixture) — keep as given
        out.append((rel, int(obj["line"]), obj["why"]))
    return out


def _encode_records(prompts: list[str]) -> bytes:
    """Length-prefixed encoding matching the Rust `decontam` stdin protocol."""
    parts: list[bytes] = [f"{len(prompts)}\n".encode("utf-8")]
    for p in prompts:
        b = p.encode("utf-8")
        parts.append(f"{len(b)}\n".encode("utf-8"))
        parts.append(b)
        parts.append(b"\n")
    return b"".join(parts)


def decontam_near(
    train: list[str], eval_prompts: list[str], *, k: int, jaccard: float
) -> list[tuple[float, str, str]]:
    """Full-coverage content-shingle near-dup scan (no eval cap). Returns
    (jaccard, train_prefix, eval_prefix) for each flagged train prompt — the same
    shape tools/assert_decontam.py builds in its `near` list."""
    header = f"{k}\n{jaccard}\n".encode("utf-8")
    stdin = header + _encode_records(train) + _encode_records(eval_prompts)
    proc = _run(["decontam"], stdin=stdin)
    if proc.returncode != 0:
        raise LexUnavailable(
            f"decontam exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:500]}"
        )
    out: list[tuple[float, str, str]] = []
    for raw in proc.stdout.decode("utf-8", "replace").splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        out.append((float(obj["j"]), obj["train"], obj["eval"]))
    return out
