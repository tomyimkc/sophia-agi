#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Diagnose GITHUB_TOKEN permissions for releases."""

from __future__ import annotations

import json
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
            print(f"  scopes: {resp.headers.get('x-oauth-scopes', '(fine-grained)')}")
    except urllib.error.HTTPError as exc:
        print(f"FAIL {exc.code} {url}")
        print(f"  needed: {exc.headers.get('x-accepted-github-permissions', '')}")
        print(f"  body: {exc.read().decode()[:300]}")


def main() -> int:
    token = load_token()
    get("https://api.github.com/user", token)
    get("https://api.github.com/repos/tomyimkc/sophia-agi", token)
    get("https://api.github.com/repos/tomyimkc/sophia-agi/releases/tags/v0.5.3", token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())