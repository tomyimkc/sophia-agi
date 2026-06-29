#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fetch a small license-clean OpenAssistant OASST1 general-instruct slice.

The slice is intentionally external to the Sophia corpus, used only for general
instruction-retention so a local Sophia adapter does not collapse into a narrow
provenance/refusal style. OASST1 declares Apache-2.0 in its Hugging Face dataset
card. This script is opt-in/networked; CI consumes the checked-in small slice.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "training" / "local_sophia_v2" / "general_instruct.jsonl"
LICENSE_OUT = ROOT / "training" / "local_sophia_v2" / "GENERAL_INSTRUCT_LICENSE.md"
TREE_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"
CARD_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/raw/main/README.md"
SOURCE_NAME = "OpenAssistant/oasst1"

SYSTEM = (
    "You are a helpful general-purpose assistant. Follow the user's instruction directly, "
    "be concise when appropriate, and do not invent sources."
)

_DENY = re.compile(
    r"\b(kill|suicide|bomb|weapon|explosive|malware|hack|porn|sex|racial slur|doxx|credit card|password)\b",
    re.I,
)


def _ok_text(text: str, *, min_len: int, max_len: int) -> bool:
    text = (text or "").strip()
    return min_len <= len(text) <= max_len and not _DENY.search(text)


def _best_reply(prompt: dict[str, Any]) -> dict[str, Any] | None:
    replies = [r for r in prompt.get("replies", []) if r.get("role") == "assistant" and r.get("review_result")]
    replies = [r for r in replies if _ok_text(r.get("text", ""), min_len=80, max_len=1600)]
    if not replies:
        return None
    return max(replies, key=lambda r: (r.get("rank", 999) * -1, r.get("review_count", 0), r.get("emojis", {}).get("+1", 0)))


def fetch(limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with urllib.request.urlopen(TREE_URL, timeout=60) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            for raw in gz:
                tree = json.loads(raw)
                prompt = tree.get("prompt") or {}
                if prompt.get("lang") != "en" or not prompt.get("review_result"):
                    continue
                user = (prompt.get("text") or "").strip()
                if not _ok_text(user, min_len=20, max_len=650):
                    continue
                reply = _best_reply(prompt)
                if reply is None:
                    continue
                rows.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": reply["text"].strip()},
                    ],
                    "metadata": {
                        "source": SOURCE_NAME,
                        "sourceUrl": TREE_URL,
                        "license": "Apache-2.0",
                        "licenseUrl": "https://huggingface.co/datasets/OpenAssistant/oasst1",
                        "messageTreeId": tree.get("message_tree_id"),
                        "promptMessageId": prompt.get("message_id"),
                        "replyMessageId": reply.get("message_id"),
                        "purpose": "general_instruction_retention",
                        "notFromSophiaCorpus": True,
                    },
                })
                if len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_license(path: Path, *, rows: int) -> None:
    path.write_text(
        "# General instruction-retention slice license\n\n"
        f"Source dataset: `{SOURCE_NAME}`.\n\n"
        "Purpose: small external general-instruct retention slice for local Sophia training; "
        "not synthesized from the Sophia corpus.\n\n"
        "Dataset card declares `license: apache-2.0`. Verify before release or commercial reuse.\n\n"
        f"Rows included: {rows}.\n\n"
        f"Dataset card: {CARD_URL}\n"
        f"Data file: {TREE_URL}\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    rows = fetch(args.limit)
    if len(rows) < args.limit:
        print(f"::error:: only fetched {len(rows)} rows; requested {args.limit}")
        return 1
    write_jsonl(args.out, rows)
    write_license(LICENSE_OUT, rows=len(rows))
    print(json.dumps({"rows": len(rows), "out": str(args.out), "license": "Apache-2.0", "source": SOURCE_NAME}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
