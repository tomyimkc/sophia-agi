#!/usr/bin/env python3
"""Create good-first GitHub issues via REST API.

Requires GITHUB_TOKEN in .env (repo scope) or environment.

Usage:
  python tools/create_github_issues.py --dry-run
  python tools/create_github_issues.py --create
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "tomyimkc/sophia-agi"

ISSUES = [
    {
        "title": "[GF-10] Define 5 more psychology concepts with subfield tags",
        "labels": ["good first issue", "corpus", "psychology"],
        "body": """## Task
Add 5 records to `data/psychology_concepts.json` with explicit `subfield` (`cognitive`, `clinical`, `pop_myth`).

## Acceptance
- [ ] Each record has `doNotAttributeTo` where applicable
- [ ] One training example or benchmark trap reference
- [ ] `python tools/validate_attribution.py` passes

See GOOD_FIRST_ISSUES.md GF-10.""",
    },
    {
        "title": "[GF-20] Add 3 dated history events with primary source field",
        "labels": ["good first issue", "corpus", "history"],
        "body": """## Task
Extend `data/history_events.json` with 3 events including `primarySource` and myth-trap notes.

## Acceptance
- [ ] Linked dispute note in `docs/04-Disputes/` or domain doc
- [ ] Benchmark case or training example hook

See GOOD_FIRST_ISSUES.md GF-20.""",
    },
    {
        "title": "[GF-30] Add scripture attribution with sect boundaries",
        "labels": ["good first issue", "corpus", "religion"],
        "body": """## Task
Add one `data/religion_concepts.json` record with tradition ids and `doNotMergeWith` boundaries.

## Acceptance
- [ ] Council-format training example or benchmark case
- [ ] Sensitive-topic handling documented

See GOOD_FIRST_ISSUES.md GF-30.""",
    },
    {
        "title": "[GF-40] Improve benchmark scorer multilingual markers",
        "labels": ["good first issue", "tooling"],
        "body": """## Task
Extend `tools/score_benchmark.py` with additional 中文 denial/affirmation patterns from failed model runs.

## Acceptance
- [ ] Regression test or documented before/after on a known failure
- [ ] No false positives on reference teacher 100% runs

See GOOD_FIRST_ISSUES.md GF-40.""",
    },
    {
        "title": "[Benchmark] Submit GPT-4o / Grok / local model leaderboard run",
        "labels": ["benchmark", "community"],
        "body": """## How
```bash
python tools/run_external_models.py --all
python tools/update_leaderboards.py
```

Or open PR with `benchmark/model_runs/*.json` + `*.report.json`.

## Need
Direct API keys: `OPENAI_API_KEY`, `XAI_API_KEY` (Monica gateway balance insufficient).""",
    },
]


def load_token() -> str | None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GITHUB_TOKEN=") or line.startswith("GH_TOKEN="):
                _, _, value = line.partition("=")
                value = value.strip().strip('"').strip("'")
                if value and "your" not in value.lower():
                    return value
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def create_issue(token: str, issue: dict) -> dict:
    body = json.dumps({"title": issue["title"], "body": issue["body"], "labels": issue.get("labels", [])}).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/issues",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "sophia-agi-create-issues",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()
    dry = not args.create

    token = load_token()
    if not dry and not token:
        print("Set GITHUB_TOKEN in .env (repo scope) to create issues.")
        return 1

    for issue in ISSUES:
        if dry:
            print(f"[dry-run] {issue['title']}")
            continue
        try:
            result = create_issue(token, issue)
            print(f"Created #{result['number']}: {result['html_url']}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            print(f"Failed {issue['title']}: HTTP {exc.code} {detail}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())