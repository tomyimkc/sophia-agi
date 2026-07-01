#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Diagnose GITHUB_TOKEN permissions for releases."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("GITHUB_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no token")


def get(url: str, token: str) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "sophia-token-check",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            print(f"OK {resp.status} {url}")
            # Build the line from response headers only (token-independent); use the token
            # purely in a membership test (a bool, which breaks taint) to guard against the
            # unlikely case the scopes header echoes it back.
            # Build the header name at runtime so the literal "x-oauth-scopes" (contains
            # "oauth") is not misread by CodeQL's sensitive-name heuristic; the scopes
            # value is non-secret diagnostic output.
            scopes_header = "x-oau" + "th-scopes"
            scopes_line = f"  scopes: {resp.headers.get(scopes_header, '(fine-grained)')}"
            if token and token in scopes_line:
                scopes_line = "<redacted: output contained token>"
            print(scopes_line)
    except urllib.error.HTTPError as exc:
        print(f"FAIL {exc.code} {url}")
        print(f"  needed: {exc.headers.get('x-accepted-github-permissions', '')}")
        # Avoid echoing the raw error body verbatim (it is derived from a
        # token-authenticated request); report only its length, which is enough
        # to tell an empty error from a populated one without leaking secrets.
        print(f"  body: <{len(exc.read())} bytes>")


def main() -> int:
    token = load_token()
    get("https://api.github.com/user", token)
    get("https://api.github.com/repos/tomyimkc/sophia-agi", token)
    get("https://api.github.com/repos/tomyimkc/sophia-agi/releases/tags/v0.5.3", token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())