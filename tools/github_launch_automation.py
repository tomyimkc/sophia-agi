#!/usr/bin/env python3
"""One-shot GitHub launch automation (About, topics, issue comments)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "tomyimkc/sophia-agi"

TOPICS = [
    "llm",
    "benchmark",
    "rag",
    "philosophy",
    "machine-learning",
    "dataset",
    "chinese",
    "attribution",
    "open-source",
    "nlp",
]

LAUNCH_COMMENT = """<!-- Launch bot -->
Thanks for picking this up! Quick links:

- Thesis: https://tomyimkc.github.io/sophia-agi/
- Run validation: `python tools/validate_attribution.py`
- See `GOOD_FIRST_ISSUES.md` for acceptance criteria

Comment here when you open a PR and we will review."""


def load_token() -> str:
    env_path = ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GITHUB_TOKEN="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SystemExit("GITHUB_TOKEN missing in .env")


def api(token: str, method: str, url: str, body: dict | None = None) -> tuple[int, dict | list]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "sophia-agi-github-automation",
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
        if isinstance(parsed, dict):
            parsed["needed"] = exc.headers.get("x-accepted-github-permissions")
        return exc.code, parsed


def main() -> int:
    token = load_token()
    base = f"https://api.github.com/repos/{REPO}"

    steps: list[tuple[str, int, str]] = []

    status, data = api(
        token,
        "PATCH",
        base,
        {
            "description": "Open corpus + benchmark for provenance-aware LLM answers (source discipline)",
            "homepage": "https://tomyimkc.github.io/sophia-agi/",
        },
    )
    detail = (data.get("description") if isinstance(data, dict) else str(data)) or str(data)[:120]
    steps.append(("PATCH repo About", status, detail))

    status, data = api(token, "PUT", f"{base}/topics", {"names": TOPICS})
    if status == 200 and isinstance(data, dict):
        detail = ", ".join(data.get("names", []))
    else:
        detail = str(data)[:160]
    steps.append(("PUT repo topics", status, detail))

    status, data = api(token, "GET", f"{base}/issues?state=open&per_page=30")
    issues: list[dict] = data if isinstance(data, list) else []
    steps.append(("LIST open issues", status, f"{len(issues)} issues"))

    for issue in issues:
        number = issue["number"]
        if number == 1:
            continue
        cstatus, comments = api(token, "GET", f"{base}/issues/{number}/comments")
        if cstatus != 200:
            steps.append((f"comments #{number}", cstatus, str(comments)[:120]))
            continue
        if isinstance(comments, list) and any("Launch bot" in (c.get("body") or "") for c in comments):
            steps.append((f"comment #{number}", 200, "already present"))
            continue
        s, created = api(token, "POST", f"{base}/issues/{number}/comments", {"body": LAUNCH_COMMENT})
        steps.append((f"comment #{number}", s, "posted" if s == 201 else str(created)[:120]))

    has_launch = any("Launch" in (i.get("title") or "") and "checklist" in (i.get("title") or "").lower() for i in issues)
    if not has_launch:
        s, created = api(
            token,
            "POST",
            f"{base}/issues",
            {
                "title": "[Launch] Public visibility checklist (Jun 2026)",
                "labels": ["documentation"],
                "body": """## Launch checklist

- [x] GitHub Pages live: https://tomyimkc.github.io/sophia-agi/
- [x] Good-first issues GF-10–GF-40 open
- [ ] Reddit r/LocalLLaMA post (`docs/07-Growth/launch/REDDIT-POST-NOW.md`)
- [ ] Revoke any PATs shared in chat; rotate token
- [ ] External benchmark runs (GPT-4o / Grok) — needs `OPENAI_API_KEY` / `XAI_API_KEY` in `.env`
- [ ] Show HN (account blocked — see `HN-BLOCKED-ALTERNATIVES.md`)

Track metrics weekly.""",
            },
        )
        url = created.get("html_url", str(created)[:120]) if isinstance(created, dict) else str(created)
        steps.append(("create launch checklist issue", s, url))

    for name, status, detail in steps:
        mark = "OK" if 200 <= status < 300 else "FAIL"
        print(f"[{mark}] {name} ({status}) — {detail}")

    failed = [s for s in steps if s[1] < 200 or s[1] >= 300]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())