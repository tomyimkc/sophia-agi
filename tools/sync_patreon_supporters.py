#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""
Patreon supporter sync for Sophia AGI.

Fetches active patrons from the Patreon API v2 using a Creator Access Token
and syncs public names into:

- SPONSORS.md (replaces the block between PATREON_SUPPORTERS_START/END markers)
- data/patreon/supporters.json (machine-readable, committed)

Usage (after putting secrets in .env or environment):

    python tools/sync_patreon_supporters.py                 # print grouped markdown
    python tools/sync_patreon_supporters.py --update        # update SPONSORS.md in place
    python tools/sync_patreon_supporters.py --write-json    # also (or only) write JSON

Environment variables (or .env next to repo root):

    PATREON_CREATOR_ACCESS_TOKEN   (required for fetching)
    PATREON_CREATOR_REFRESH_TOKEN  (optional, for refresh on 401)
    PATREON_CAMPAIGN_ID            (optional, skips /campaigns discovery)
    PATREON_CLIENT_ID
    PATREON_CLIENT_SECRET          (both required only if you want auto-refresh)

The script is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPONSORS_MD = ROOT / "SPONSORS.md"
DEFAULT_JSON = ROOT / "data" / "patreon" / "supporters.json"
TIERS_CONFIG = ROOT / "data" / "patreon" / "tiers.json"

PATREON_BASE = "https://www.patreon.com/api/oauth2/v2"
MARKER_START = "<!-- PATREON_SUPPORTERS_START -->"
MARKER_END = "<!-- PATREON_SUPPORTERS_END -->"


def load_tiers_config() -> dict:
    """Load canonical Patreon tier definitions (for ordering + nice names)."""
    if TIERS_CONFIG.exists():
        try:
            return json.loads(TIERS_CONFIG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"tiers": [], "ordering": []}


def get_tier_display_order() -> list[str]:
    cfg = load_tiers_config()
    order = cfg.get("ordering", [])
    # Return the patreon_title values in the defined order
    title_by_key = {t["key"]: t["patreon_title"] for t in cfg.get("tiers", [])}
    return [title_by_key.get(k, k) for k in order if k in title_by_key]


def normalize_tier_title(raw: str) -> str:
    """Try to map a Patreon tier title to the canonical one from config."""
    if not raw:
        return "Supporter"
    cfg = load_tiers_config()
    titles = {t["patreon_title"]: t["patreon_title"] for t in cfg.get("tiers", [])}
    # Also allow matching by key or partial
    for t in cfg.get("tiers", []):
        if raw == t["patreon_title"] or raw == t["key"]:
            return t["patreon_title"]
        if raw.lower() in t["patreon_title"].lower() or t["patreon_title"].lower() in raw.lower():
            return t["patreon_title"]
    return raw  # fall back to whatever Patreon gave us


def load_dotenv_simple(env_path: Path) -> None:
    """Very small .env loader (KEY=val, ignores comments and quotes). No deps."""
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # Only set if not already in environment (env wins)
        if key and key not in os.environ:
            os.environ[key] = val


def get_env(name: str, required: bool = False) -> str | None:
    val = os.environ.get(name)
    if required and not val:
        print(f"ERROR: Missing required environment variable: {name}", file=sys.stderr)
        print("       Set it in .env or export before running.", file=sys.stderr)
        sys.exit(2)
    return val


def patreon_request(
    path: str,
    token: str,
    params: dict[str, Any] | None = None,
    method: str = "GET",
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal Patreon API caller using stdlib urllib."""
    url = PATREON_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "sophia-agi-patreon-sync/1.0 (+https://github.com/tomyimkc/sophia-agi)",
    }

    body = None
    if method == "POST" and data:
        body = urllib.parse.urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Patreon API error {e.code}: {err_body}") from e


def refresh_access_token(client_id: str, client_secret: str, refresh_token: str) -> dict[str, Any]:
    """Refresh using the refresh token. Returns new token dict."""
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }
    # This endpoint is special and does not use /v2
    url = "https://www.patreon.com/api/oauth2/token"
    body = urllib.parse.urlencode(data).encode("utf-8")
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to refresh Patreon token ({e.code}): {err_body}") from e


def discover_campaign_id(token: str) -> str:
    """Return the first campaign id for the authenticated creator."""
    params = {"fields[campaign]": "vanity,summary,creation_name"}
    resp = patreon_request("/campaigns", token, params)
    data = resp.get("data", [])
    if not data:
        raise RuntimeError("No campaigns found for this creator token.")
    return data[0]["id"]


def fetch_active_members(token: str, campaign_id: str) -> list[dict[str, Any]]:
    """
    Return list of active patrons with their tier titles.

    We request:
      - member: full_name, patron_status, ...
      - include currently_entitled_tiers
      - tier: title
    """
    all_members: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "fields[member]": "full_name,patron_status,currently_entitled_amount_cents",
        "include": "currently_entitled_tiers",
        "fields[tier]": "title,amount_cents",
        "page[count]": "100",
    }

    page = 0

    while True:
        page += 1
        if page > 20:  # safety
            break

        resp = patreon_request(f"/campaigns/{campaign_id}/members", token, params)

        included = {inc["id"]: inc for inc in resp.get("included", []) if inc.get("type") == "tier"}

        for m in resp.get("data", []):
            attrs = m.get("attributes", {})
            status = attrs.get("patron_status")
            if status != "active_patron":
                continue

            name = (attrs.get("full_name") or "").strip()
            if not name:
                continue

            # Find entitled tiers from relationships
            tier_titles = []
            rel = m.get("relationships", {}).get("currently_entitled_tiers", {})
            tier_data = rel.get("data") or []
            for t in tier_data:
                tid = t.get("id")
                if tid and tid in included:
                    title = included[tid].get("attributes", {}).get("title", "").strip()
                    if title:
                        tier_titles.append(title)

            # If no specific tier title (rare), fall back to generic
            raw_tier = tier_titles[0] if tier_titles else "Supporter"
            tier = normalize_tier_title(raw_tier)

            all_members.append({
                "name": name,
                "tier": tier,
                "amount_cents": attrs.get("currently_entitled_amount_cents") or 0,
            })

        # Pagination
        links = resp.get("links", {})
        next_link = links.get("next")
        if not next_link:
            break

        # Extract cursor if present
        if isinstance(next_link, str) and "page[cursor]" in next_link:
            parsed = urllib.parse.urlparse(next_link)
            qs = urllib.parse.parse_qs(parsed.query)
            cursor = qs.get("page[cursor]", [None])[0]
            if cursor:
                params = {**params, "page[cursor]": cursor}
                continue

        # If no cursor style next, we stop (should not normally happen)
        break

    # Deduplicate by name (keep highest amount in case of multi-tier edge cases)
    by_name: dict[str, dict[str, Any]] = {}
    for m in all_members:
        existing = by_name.get(m["name"])
        if not existing or m["amount_cents"] > existing["amount_cents"]:
            by_name[m["name"]] = m

    # Sort within tiers later; return list sorted by amount desc overall
    result = list(by_name.values())
    result.sort(key=lambda x: (-x["amount_cents"], x["name"].lower()))
    return result


def group_by_tier(members: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for m in members:
        tier = m["tier"] or "Supporter"
        groups.setdefault(tier, []).append(m["name"])
    # Sort names inside each tier
    for tier in groups:
        groups[tier].sort(key=str.lower)
    return groups


def build_markdown(groups: dict[str, list[str]], synced_at: str) -> str:
    if not groups:
        return "_No active Patreon supporters synced yet._"

    lines: list[str] = []

    # Use canonical order from data/patreon/tiers.json when available
    canonical_order = get_tier_display_order()
    ordered_tiers: list[str] = []

    # First add tiers that exist in canonical order
    for t in canonical_order:
        if t in groups:
            ordered_tiers.append(t)

    # Then append any remaining tiers (new or unmapped) in alpha order
    for t in sorted(groups.keys()):
        if t not in ordered_tiers:
            ordered_tiers.append(t)

    for tier in ordered_tiers:
        names = groups[tier]
        if not names:
            continue
        lines.append(f"**{tier}**")
        for name in names:
            lines.append(f"- {name}")
        lines.append("")  # blank line between tiers

    header = (
        f"<!-- Last sync: {synced_at} (UTC) via tools/sync_patreon_supporters.py -->\n"
        f"<!-- AUTO-GENERATED — do not edit by hand; run `python tools/sync_patreon_supporters.py --update` -->"
    )
    return header + "\n\n" + "\n".join(lines).strip()


def update_sponsors_md(markdown_block: str) -> bool:
    """Replace content between the Patreon markers. Returns True if file was changed."""
    if not SPONSORS_MD.exists():
        print(f"ERROR: {SPONSORS_MD} not found", file=sys.stderr)
        return False

    content = SPONSORS_MD.read_text(encoding="utf-8")

    if MARKER_START not in content or MARKER_END not in content:
        print("ERROR: Patreon markers not found in SPONSORS.md", file=sys.stderr)
        print(f"       Expected {MARKER_START} ... {MARKER_END}", file=sys.stderr)
        return False

    start_idx = content.find(MARKER_START)
    end_idx = content.find(MARKER_END)
    if end_idx < start_idx:
        print("ERROR: malformed markers (END before START)", file=sys.stderr)
        return False

    # Include the end marker line
    end_idx = content.find(MARKER_END) + len(MARKER_END)

    before = content[:start_idx]
    after = content[end_idx:]

    # Build the replacement block
    new_block = f"{MARKER_START}\n{markdown_block}\n{MARKER_END}"

    new_content = before + new_block + after

    if new_content == content:
        print("No changes to SPONSORS.md (already up to date).")
        return False

    SPONSORS_MD.write_text(new_content, encoding="utf-8")
    print(f"Updated {SPONSORS_MD}")
    return True


def write_json(members: list[dict[str, Any]], groups: dict[str, list[str]], out_path: Path) -> None:
    cfg = load_tiers_config()
    tier_order = get_tier_display_order()

    payload = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "source": "patreon_api_v2",
        "count": len(members),
        "tier_order": tier_order,
        "tiers": groups,
        "members": [
            {"name": m["name"], "tier": m["tier"]} for m in members
        ],
        "tiers_config_version": cfg.get("version"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Patreon active supporters into Sophia repo content.")
    parser.add_argument("--update", action="store_true", help="Update SPONSORS.md in place")
    parser.add_argument("--write-json", action="store_true", help="Write data/patreon/supporters.json")
    parser.add_argument("--json-path", default=str(DEFAULT_JSON), help="Custom JSON output path")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files, just show what would happen")
    args = parser.parse_args()

    # Load .env if present (simple, no extra package)
    load_dotenv_simple(ROOT / ".env")

    access_token = get_env("PATREON_CREATOR_ACCESS_TOKEN", required=True)
    campaign_id = get_env("PATREON_CAMPAIGN_ID")
    refresh_token = get_env("PATREON_CREATOR_REFRESH_TOKEN")
    client_id = get_env("PATREON_CLIENT_ID")
    client_secret = get_env("PATREON_CLIENT_SECRET")

    # Optional refresh attempt if we get 401 later (we do a proactive check)
    def get_valid_token(tok: str) -> str:
        # We do a cheap call to validate
        try:
            patreon_request("/identity", tok, {"fields[user]": "full_name"})
            return tok
        except RuntimeError as e:
            if "401" in str(e) and refresh_token and client_id and client_secret:
                print("Access token invalid, attempting refresh...", file=sys.stderr)
                new_tokens = refresh_access_token(client_id, client_secret, refresh_token)
                new_access = new_tokens.get("access_token")
                if new_access:
                    print("Successfully refreshed access token.", file=sys.stderr)
                    # Note: we do not persist it here. User / CI should update the secret.
                    return new_access
            raise

    try:
        token = get_valid_token(access_token)

        if not campaign_id:
            print("Discovering campaign ID...")
            campaign_id = discover_campaign_id(token)
            print(f"Using campaign: {campaign_id}")

        print("Fetching active Patreon members...")
        members = fetch_active_members(token, campaign_id)
        groups = group_by_tier(members)

        synced_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        md = build_markdown(groups, synced_at)

        print("\n=== GROUPED SUPPORTERS (Patreon) ===")
        print(md)
        print("====================================\n")

        if args.dry_run:
            print("[dry-run] Not writing any files.")
            return 0

        changed = False
        if args.update:
            changed = update_sponsors_md(md)

        if args.write_json:
            write_json(members, groups, Path(args.json_path))

        if args.update and not changed:
            print("SPONSORS.md was already in sync.")

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
