#!/usr/bin/env python3
"""Verify-then-exec shim: gate the codebase-memory binary on its sha256 pin, then exec it.

Wired in ``.mcp.json`` AFTER ``index_guard.py`` (the locked-tree gate):

    python tools/cbm/index_guard.py -- \\
        python tools/cbm/verify_then_exec.py <pinned-binary> [args...]

Runs ``fetch_cbm.verify(<binary>)`` against ``cbm.pin.json`` and REFUSES (exit 1) on any
sha256 mismatch or an uninitialized pin; otherwise ``exec``s ``<binary> [args...]`` so the
server process replaces this shim. Net: no launch path can skip the byte-pin (the shim) or
the locked-tree check (index_guard). canClaimAGI:false.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.cbm.fetch_cbm import load_pin, verify  # noqa: E402


def main(argv: "list[str]") -> int:
    if not argv:
        sys.stderr.write("usage: verify_then_exec.py <binary> [args...]\n")
        return 2
    binary = Path(argv[0])
    ok, msg = verify(binary, load_pin())
    if not ok:
        sys.stderr.write(f"[cbm-verify] REFUSING to exec: {msg}\n")
        return 1
    sys.stderr.write(f"[cbm-verify] {msg}\n")
    os.execvp(str(binary), [str(binary), *argv[1:]])  # replaces this process on success
    return 127  # only reached if execvp itself fails


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
