#!/usr/bin/env python3
"""Deterministic git-crypt lock-state detection.

A git-crypt file begins with the 10 magic bytes b"\x00GITCRYPT\x00". Plaintext
never starts with a NUL, so a byte-exact compare of the first 10 bytes tells
LOCKED (ciphertext, safe) from UNLOCKED (plaintext) with no grep-on-binary or
`grep -z` portability footgun (both broken under macOS ugrep per the audit).
"""
from __future__ import annotations
import sys
from pathlib import Path

GIT_CRYPT_MAGIC = b"\x00GITCRYPT\x00"
CANARY = "AGENTS.md"  # a known filter=git-crypt path in this repo


def is_locked(path: str = CANARY) -> bool:
    """True if `path` is git-crypt ciphertext (tree LOCKED == safe to index/serve)."""
    try:
        head = Path(path).read_bytes()[: len(GIT_CRYPT_MAGIC)]
    except OSError:
        return False
    return head == GIT_CRYPT_MAGIC


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else CANARY
    return 0 if is_locked(path) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
