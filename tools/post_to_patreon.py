#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""
Post content from this GitHub repo to Patreon.

Uses your existing Patreon creator credentials (from .env).

Typical uses:
  - Post release notes / changelog sections
  - Post updates, new features, or sponsor thanks
  - Post tier-specific or patron-only announcements
  - Regular "supporter thank you" posts (helper mode for huge supporter lists)

The script is stdlib-only (same style as sync_patreon_supporters.py).

Usage examples:
  # Dry run (see what would be posted)
  python tools/post_to_patreon.py --title "New cluster simulator" --file CHANGELOG.md --dry-run

  # Post publicly
  python tools/post_to_patreon.py --title "Sophia v0.9 update" --content "..." --public

  # Patron-only post visible to everyone pledging >= HKD99
  python tools/post_to_patreon.py --title "For supporters" --file some-update.md --min-cents 9900

  # Helper mode: auto-generate nice thank-you post from synced supporters data
  python tools/post_to_patreon.py --supporter-post --min-cents 9900 --dry-run

Markdown is converted to basic HTML automatically.

Environment variables (put in .env):
  PATREON_CREATOR_ACCESS_TOKEN (required)

Note: unlike sync_patreon_supporters.py, this script does NOT implement token
refresh on 401. Use a valid (non-expired) creator access token. If you hit 401,
generate a fresh token on the Patreon developer portal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PATREON_BASE = "https://www.patreon.com/api/oauth2/v2"


def load_dotenv_simple(env_path: Path) -> None:
    """Very small .env loader (same as sync_patreon_supporters.py)."""
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_env(name: str, required: bool = False) -> str | None:
    val = os.environ.get(name)
    if required and not val:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(2)
    return val


def patreon_request(
    path: str,
    token: str,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal Patreon API caller (same style as the supporters script)."""
    url = PATREON_BASE + path
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "sophia-agi-patreon-post/1.0 (+https://github.com/tomyimkc/sophia-agi)",
    }

    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
            if text:
                return json.loads(text)
            return {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Patreon API error {e.code}: {err_body}") from e


def markdown_to_html(md: str) -> str:
    """
    Very basic Markdown → HTML converter (stdlib only).
    Good enough for changelogs, release notes, and simple docs.
    Supports: headers, lists, bold, italic, links, paragraphs, code spans.
    """
    if not md:
        return ""

    # Join wrapped lines inside paragraphs (helps with long changelog lines).
    # The lookahead-exclusion set must include ordered-list markers (digit + '.'),
    # otherwise a line starting with "1." would be joined onto the previous line.
    md = re.sub(r'([^\n])\n(?=[^\n#\-*+\s])(?!\d+\.\s)', r'\1 \2', md)

    lines = md.strip().splitlines()
    out: list[str] = []
    in_ul = False
    in_ol = False

    def inline(text: str) -> str:
        # code first
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # bold
        text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', text)
        # italic
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'_([^_]+)_', r'<em>\1</em>', text)
        # links
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        return text

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # blank line
        if not line:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            i += 1
            continue

        # ATX headers
        m = re.match(r'^(#{1,6})\s+(.*)$', line)
        if m:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            if in_ol:
                out.append("</ol>")
                in_ol = False
            level = len(m.group(1))
            content = inline(m.group(2).strip())
            out.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        # unordered list
        if re.match(r'^[-*+]\s+', line):
            if not in_ul:
                if in_ol:
                    out.append("</ol>")
                    in_ol = False
                out.append("<ul>")
                in_ul = True
            item = re.sub(r'^[-*+]\s+', '', line)
            out.append(f"<li>{inline(item)}</li>")
            i += 1
            continue

        # ordered list
        if re.match(r'^\d+\.\s+', line):
            if not in_ol:
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                out.append("<ol>")
                in_ol = True
            item = re.sub(r'^\d+\.\s+', '', line)
            out.append(f"<li>{inline(item)}</li>")
            i += 1
            continue

        # close lists if we hit non-list
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

        # paragraph
        out.append(f"<p>{inline(line)}</p>")
        i += 1

    if in_ul:
        out.append("</ul>")
    if in_ol:
        out.append("</ol>")

    return "\n".join(out)


def create_post(
    token: str,
    title: str,
    content_html: str,
    is_public: bool = True,
    min_cents: int | None = None,
) -> dict[str, Any]:
    """Create a post on Patreon."""
    payload: dict[str, Any] = {
        "data": {
            "type": "post",
            "attributes": {
                "title": title,
                "content": content_html,
                "is_public": is_public,
            },
        }
    }

    if min_cents is not None:
        payload["data"]["attributes"]["min_cents"] = int(min_cents)

    return patreon_request("/posts", token, method="POST", data=payload)


def build_supporter_post_content(data: dict) -> str:
    """
    Generate a nice, expandable thank-you post for Patreon.
    Designed to scale to "huge" numbers of supporters:
    - Uses tier_order from the sync data (bilingual tier names already)
    - Shows names when reasonable, falls back to counts for very large tiers
    - Includes link back to GitHub SPONSORS.md for the full list
    - Bilingual-friendly structure (English + Chinese tier names are used as-is)
    """
    tiers = data.get("tiers", {}) or {}
    tier_order = data.get("tier_order", []) or list(tiers.keys())
    total = data.get("count", 0) or sum(len(v) for v in tiers.values())
    synced = data.get("synced_at", "unknown")[:10]

    lines: list[str] = []

    lines.append("# Thank You, Patreon Supporters ❤️")
    lines.append("")
    lines.append(f"As of **{synced}**, Sophia has **{total} active Patreon supporters**.")
    lines.append("")
    lines.append(
        "Your contributions directly fund the time and compute needed to keep this project "
        "honest: fail-closed provenance gates, reproducible benchmarks, and the public failure ledger."
    )
    lines.append("")
    lines.append("Wisdom before intelligence — thank you for making it possible.")
    lines.append("")

    if tier_order and any(tiers.get(t) for t in tier_order):
        lines.append("## Supporters by Tier")
        lines.append("")

        for tier in tier_order:
            names: list[str] = tiers.get(tier, []) or []
            if not names:
                continue

            lines.append(f"### {tier}")
            n = len(names)

            if n <= 12:
                for name in names:
                    lines.append(f"- {name}")
            else:
                # For huge expansions, show a sample + count + link to full list
                sample = names[:8]
                for name in sample:
                    lines.append(f"- {name}")
                lines.append(f"- … and {n - 8} more")
                lines.append("")
                lines.append(
                    f"Full list (with GitHub Sponsors too): "
                    "[SPONSORS.md](https://github.com/tomyimkc/sophia-agi/blob/main/SPONSORS.md)"
                )

            lines.append("")

    lines.append("## Join the circle")
    lines.append("")
    lines.append("If you find the work valuable, consider supporting at any tier:")
    lines.append("https://www.patreon.com/c/aideveloper_tomyim")
    lines.append("")
    lines.append("Every pledge helps maintain the no-overclaim standard and the public proof package.")
    lines.append("")
    lines.append("— tomyimkc")

    return "\n".join(lines)


def extract_latest_changelog_section(changelog_path: Path) -> tuple[str, str]:
    """Naive extractor for the first section of CHANGELOG.md (Unreleased or latest version)."""
    if not changelog_path.exists():
        return "Update from Sophia AGI", ""

    text = changelog_path.read_text(encoding="utf-8")
    # Find first ## section after the title
    parts = re.split(r'\n(?=##\s)', text, maxsplit=2)
    if len(parts) >= 2:
        header = parts[1].split('\n', 1)[0].strip()
        body = parts[1].split('\n', 1)[1] if '\n' in parts[1] else ""
        # clean up the header for title
        title = re.sub(r'^##\s*\[?([^\]]+)\]?.*', r'\1', header).strip()
        return title or "Sophia AGI Update", body.strip()
    return "Sophia AGI Update", text.strip()[:2000]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post GitHub content (Markdown) to Patreon using creator token."
    )
    parser.add_argument("--title", help="Post title (auto-generated for --supporter-post if omitted)")
    parser.add_argument("--content", help="Raw content (Markdown)")
    parser.add_argument("--file", type=Path, help="Read content from this Markdown file")
    parser.add_argument("--changelog", action="store_true", help="Use top section of CHANGELOG.md as content")
    parser.add_argument("--public", action="store_true", help="Make the post public (default: patrons only)")
    parser.add_argument("--min-cents", type=int, help="Minimum pledge in cents to see the post (e.g. 9900 for HKD99)")
    parser.add_argument("--dry-run", action="store_true", help="Print payload instead of posting")
    parser.add_argument("--env", default=str(ROOT / ".env"), help="Path to .env file")

    # Helper mode for large supporter bases
    parser.add_argument("--supporter-post", action="store_true",
                        help="Auto-generate a bilingual-friendly thank-you post from data/patreon/supporters.json (ideal for huge supporter page expansion)")

    args = parser.parse_args()

    load_dotenv_simple(Path(args.env))

    token = get_env("PATREON_CREATOR_ACCESS_TOKEN", required=True)

    # === Special helper mode for huge supporter expansions ===
    if args.supporter_post:
        supporters_path = ROOT / "data" / "patreon" / "supporters.json"
        if not supporters_path.exists():
            print("ERROR: No supporters data found. Run `python tools/sync_patreon_supporters.py --update --write-json` first.", file=sys.stderr)
            return 1

        data = json.loads(supporters_path.read_text(encoding="utf-8"))
        md_content = build_supporter_post_content(data)

        if not args.title:
            date_str = datetime.now().strftime("%Y-%m")
            args.title = f"Thank You to Our Patreon Supporters — {date_str}"

        # Sensible defaults for supporter posts (patron-only, no strict min unless user sets --min-cents)
        if args.min_cents is None and not args.public:
            # Leave min_cents as None → visible to all active patrons
            pass

        print("[supporter-post helper] Generated thank-you content from synced data.")
    else:
        # Normal content collection
        md_content = None
        if args.changelog:
            ch_title, md_content = extract_latest_changelog_section(ROOT / "CHANGELOG.md")
            if not args.title:
                args.title = ch_title
        elif args.file:
            md_content = args.file.read_text(encoding="utf-8")
        elif args.content:
            md_content = args.content
        else:
            print("ERROR: provide --content, --file, --changelog, or --supporter-post", file=sys.stderr)
            return 1

        if md_content is None:
            print("ERROR: no content collected", file=sys.stderr)
            return 1

    if not args.title:
        print("ERROR: --title is required (it can be auto-generated by --supporter-post or --changelog)", file=sys.stderr)
        return 1

    html_content = markdown_to_html(md_content)

    is_public = bool(args.public)
    min_cents = args.min_cents

    print("=== Patreon Post Preview ===")
    print(f"Title: {args.title}")
    print(f"Public: {is_public}")
    if min_cents:
        print(f"Min pledge (cents): {min_cents}")
    print(f"Content length (HTML): {len(html_content)} chars")
    print("--- First 800 chars of HTML ---")
    print(html_content[:800] + ("..." if len(html_content) > 800 else ""))
    print("==============================")

    if args.dry_run:
        print("\n[DRY RUN] Not posting. Remove --dry-run to publish.")
        return 0

    try:
        resp = create_post(
            token=token,
            title=args.title,
            content_html=html_content,
            is_public=is_public,
            min_cents=min_cents,
        )
        post_id = resp.get("data", {}).get("id", "unknown")
        print(f"\n✅ Posted successfully! Post ID: {post_id}")
        print("You can view/edit it in your Patreon creator dashboard.")
        return 0
    except Exception as exc:
        print(f"\nERROR posting: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
