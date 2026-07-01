#!/usr/bin/env python3
"""L2 sink guard: fail if any codebase-memory index artifact is staged/tracked.

The graph cache/artifact embeds source-derived docstrings/signatures and is
git-crypt-blind, so committing it would leak. This reads the git index/tree
directly (survives `git add -f` and a deleted .gitignore line) and flags:
  * path      .codebase-memory/**, graph.db*, *.db.zst
  * magic     zstd (28 B5 2F FD) or SQLite ("SQLite format 3\\0") on the
              staged/incremental set, any path (catches a renamed artifact)
Pure Python — no `grep -z` (ugrep treats -z as decompress).
"""
from __future__ import annotations
import re
import subprocess
import sys

SINK_RE = re.compile(r"(^|/)(\.codebase-memory/|graph\.db($|\.|[^a-zA-Z0-9])|[^/]*\.db\.zst$)")
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
SQLITE_MAGIC = b"SQLite format 3\x00"


def _paths(staged_only: bool, cwd: str) -> list[str]:
    cmd = (["git", "diff", "--cached", "--name-only", "-z"] if staged_only
           else ["git", "ls-files", "-z"])
    out = subprocess.run(cmd, cwd=cwd, capture_output=True, check=True).stdout
    return [p.decode("utf-8", "surrogateescape") for p in out.split(b"\x00") if p]


def _blob_head(path: str, staged_only: bool, cwd: str, n: int = 16) -> bytes:
    ref = f":{path}" if staged_only else f"HEAD:{path}"
    r = subprocess.run(["git", "show", ref], cwd=cwd, capture_output=True)
    return r.stdout[:n] if r.returncode == 0 else b""


def find_violations(staged_only: bool, cwd: str = ".") -> list[str]:
    viol: list[str] = []
    for path in _paths(staged_only, cwd):
        if SINK_RE.search(path):
            viol.append(f"{path} (sink path)")
            continue
        # Magic-byte backstop for a renamed/relocated artifact. Runs on the
        # STAGED (incremental) set only — where such an artifact first enters —
        # keeping it cheap and false-positive-safe. The tracked/CI scan is
        # PATH-ONLY (no magic-byte scan): the path rules cover every known
        # artifact name, and the load-bearing public-exposure guarantee does
        # NOT depend on magic-byte detection at the CI/publish layer; it is
        # carried by L1 (build refused on an unlocked tree, so no plaintext
        # secret can ever be indexed) + L3 (no git-crypt key in CI, so the CI
        # checkout is ciphertext noise). Do NOT expand this block into an
        # expensive full-tree magic scan in tracked mode — that would add cost
        # and false positives without strengthening the actual guarantee.
        if staged_only:
            head = _blob_head(path, staged_only, cwd)
            if head.startswith(ZSTD_MAGIC) or head.startswith(SQLITE_MAGIC):
                viol.append(f"{path} (index blob magic bytes)")
    return viol


def main(argv: list[str]) -> int:
    staged_only = "--staged" in argv
    viol = find_violations(staged_only)
    if viol:
        sys.stderr.write("ERROR: a codebase-memory index artifact must never be committed:\n")
        for v in viol:
            sys.stderr.write(f"  - {v}\n")
        sys.stderr.write("It embeds source-derived docstrings/signatures. Remove it before committing.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
