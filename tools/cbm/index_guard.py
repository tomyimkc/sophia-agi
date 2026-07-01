#!/usr/bin/env python3
"""L1 locked-tree preflight — the ONLY entrypoint to the codebase-memory binary.

Aborts (before any DB write / before the MCP server binds) unless the working
tree is git-crypt LOCKED and no git-crypt key is present. Wire as the .mcp.json
`command` so a raw invocation can't bypass it.

Usage: python tools/cbm/index_guard.py -- <binary> [args...]

Known limitation (residual risk R1): a startup check does not cover the tree
being unlocked MID-session while the server runs; the public-exposure guarantee
still holds via L2 (sink) + L3 (no key in CI). Never build a shareable artifact
from a session that was unlocked mid-run.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

# Allow running as a script (`python tools/cbm/index_guard.py ...`): put the repo
# root on sys.path so the import below resolves in BOTH script and `python -m` mode.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.cbm.lockcheck import is_locked  # noqa: E402


def preflight(env: dict, locked: bool) -> str | None:
    if env.get("GITCRYPT_KEY_B64"):
        return "GITCRYPT_KEY_B64 is set — refusing (tree may be unlocked)."
    if not locked:
        return "working tree is UNLOCKED (AGENTS.md is plaintext) — refusing to index/serve."
    return None


def main(argv: list[str]) -> int:
    if "--" not in argv:
        sys.stderr.write("usage: index_guard.py -- <binary> [args...]\n")
        return 2
    cmd = argv[argv.index("--") + 1:]
    if not cmd:
        sys.stderr.write("usage: index_guard.py -- <binary> [args...]\n")
        return 2
    err = preflight(dict(os.environ), is_locked())
    if err:
        sys.stderr.write(f"[cbm-guard] {err}\n")
        return 1
    os.execvp(cmd[0], cmd)  # replaces the process; returns only on failure
    return 127


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
