#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""System-prompt hygiene gate (OWASP LLM07).

Scan the repo's *public* prompt-bearing surfaces for things that must never ship
inside a system prompt or instruction string: API keys, private keys, internal
hostnames/IPs, developer home paths, and live canary tokens. Fail closed (exit 1)
on any hit so a secret can never reach a published prompt.

Scope: only files that are PUBLIC in the repo. git-crypt-encrypted prompts
(AGENTS.md, CONTRACT.md, .claude/skills/**) ship as ciphertext, so a secret there
is not leaked; this gate deliberately skips them.

    python tools/check_prompt_hygiene.py            # scan the default surfaces
    python tools/check_prompt_hygiene.py --json
    python tools/check_prompt_hygiene.py path/to/extra.py ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.canary import scan_for_canaries  # noqa: E402
from agent.secret_patterns import find_internal, find_secrets  # noqa: E402

# Public surfaces that carry model-facing instruction text. Globs are resolved
# relative to the repo root. Keep this list in sync with where prompts live.
DEFAULT_GLOBS = [
    "gateway/server.py",
    "gateway/*.py",
    "sophia_mcp/server.py",
    "sophia_mcp/*.py",
    "constitution/*.json",
    "agent/*prompt*.py",
    "skills/**/*.json",
    "skills/**/*.md",
    "prompts/**/*",
]


def _git_crypt_encrypted(path: Path) -> bool:
    """True if the file is a git-crypt blob (so its plaintext is not public)."""
    try:
        with path.open("rb") as fh:
            head = fh.read(10)
        return head.startswith(b"\x00GITCRYPT")
    except OSError:
        return True  # unreadable → treat as out of scope, fail-safe to skip


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)  # outside the repo tree (e.g. an explicit path / tmp)


def _tracked(path: Path) -> bool:
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return False  # outside the repo is, by definition, not tracked here
    try:
        out = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)],
                             cwd=ROOT, capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return True  # if git is unavailable, scan it anyway


def collect_files(globs: "list[str]") -> "list[Path]":
    seen: dict[Path, None] = {}
    for pattern in globs:
        p = Path(pattern)
        # An explicit existing file (absolute or relative) is taken as-is; only
        # repo-relative glob patterns go through ROOT.glob (which rejects absolute
        # patterns). This lets callers pass concrete paths outside the tree.
        if p.is_absolute() or p.exists():
            if p.is_file():
                seen.setdefault(p.resolve(), None)
            continue
        for hit in ROOT.glob(pattern):
            if hit.is_file():
                seen.setdefault(hit, None)
    return sorted(seen)


def _mask(value: str) -> str:
    """Describe a finding WITHOUT echoing the secret itself.

    A security scanner must never print the matched secret (clear-text logging of
    sensitive data). We surface only the length, which is enough to locate it
    while leaking nothing.
    """
    return f"<redacted: {len(value)} chars>"


def scan_file(path: Path) -> "list[dict]":
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [{"kind": "unreadable", "preview": str(exc)}]
    findings: list[dict] = []
    for f in find_secrets(text):
        findings.append({"kind": f"secret:{f['kind']}", "preview": _mask(f["match"])})
    for f in find_internal(text):
        findings.append({"kind": f"internal:{f['kind']}", "preview": _mask(f["match"])})
    for c in scan_for_canaries(text):
        findings.append({"kind": "canary", "preview": _mask(c)})
    return findings


def run(globs: "list[str]") -> dict:
    results: list[dict] = []
    for path in collect_files(globs):
        if not _tracked(path) or _git_crypt_encrypted(path):
            continue
        hits = scan_file(path)
        if hits:
            results.append({"file": _relpath(path), "findings": hits})
    return {"clean": not results, "files_with_findings": results}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="extra files/globs to scan (added to defaults)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    report = run(DEFAULT_GLOBS + list(args.paths))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif report["clean"]:
        print("prompt hygiene: OK — no secrets/internal identifiers/canaries in public prompts")
    else:
        print("prompt hygiene: FAIL — secrets or internal identifiers found in public prompts:")
        for entry in report["files_with_findings"]:
            print(f"  {entry['file']}")
            for f in entry["findings"]:
                print(f"    - {f['kind']}: {f.get('preview', '')}")
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
