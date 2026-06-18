#!/usr/bin/env python3
"""Create or update a GitHub Release from CHANGELOG.md section.

Usage:
  python tools/create_github_release.py --tag v0.5.3
  python tools/create_github_release.py --tag v0.5.3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "tomyimkc/sophia-agi"
CHANGELOG = ROOT / "CHANGELOG.md"


def load_token() -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise SystemExit("Missing .env with GITHUB_TOKEN")
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GITHUB_TOKEN="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value and "your" not in value.lower():
                return value
    raise SystemExit("GITHUB_TOKEN missing in .env")


def extract_notes(tag: str) -> str:
    version = tag.lstrip("v")
    text = CHANGELOG.read_text(encoding="utf-8")
    pattern = rf"## \[{re.escape(version)}\][^\n]*\n(.*?)(?=\n## \[|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise SystemExit(f"No CHANGELOG section for {tag}")
    body = match.group(1).strip()
    return f"# Sophia AGI {tag}\n\n{body}\n\n**Links:** [Thesis](https://tomyimkc.github.io/sophia-agi/) · [HF corpus](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus)\n"


def api(token: str, method: str, url: str, body: dict | None = None) -> tuple[int, dict | list]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "sophia-agi-release",
    }
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode()
        try:
            parsed: dict | list = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {"error": payload}
        return exc.code, parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Create GitHub release from CHANGELOG")
    parser.add_argument("--tag", default=f"v{(ROOT / 'VERSION').read_text(encoding='utf-8').strip()}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    notes = extract_notes(args.tag)
    if args.dry_run:
        print(notes)
        return 0

    token = load_token()
    base = f"https://api.github.com/repos/{REPO}/releases"
    status, existing = api(token, "GET", f"{base}/tags/{args.tag}")
    payload = {
        "tag_name": args.tag,
        "name": f"Sophia AGI {args.tag}",
        "body": notes,
        "draft": False,
        "prerelease": False,
    }
    if status == 200 and isinstance(existing, dict) and existing.get("id"):
        release_id = existing["id"]
        status, data = api(token, "PATCH", f"{base}/{release_id}", payload)
        action = "Updated"
    else:
        status, data = api(token, "POST", base, payload)
        action = "Created"

    if status not in (200, 201):
        print(f"Release failed ({status}): {json.dumps(data, indent=2)}")
        return 1

    url = data.get("html_url") if isinstance(data, dict) else ""
    print(f"{action} release: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())