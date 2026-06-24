#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Librarian CLI: ingest a raw source into a gated OKF wiki draft.

    python tools/wiki_ingest.py raw/sample-oikeiosis.txt --provider mock
    python tools/wiki_ingest.py raw/my-source.txt --provider glm --tier draft

Reads the source as untrusted text, asks the model (via the unified adapter) for a
single provenance-stamped page, and writes it only if it passes the source-discipline
gate. Offline-testable with --provider mock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import wiki_librarian  # noqa: E402
from agent.model import default_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a raw source into the OKF wiki (gated draft)")
    parser.add_argument("source", type=Path, help="path to a raw source file")
    parser.add_argument("--provider", default=None, help="model provider preset (e.g. mock, glm, anthropic)")
    parser.add_argument("--tier", default="draft", choices=["draft", "memory"], help="target tier")
    args = parser.parse_args()

    if not args.source.exists():
        print(json.dumps({"error": f"source not found: {args.source}"}, indent=2))
        return 1

    text = args.source.read_text(encoding="utf-8")
    client = default_client(args.provider) if args.provider else default_client()
    result = wiki_librarian.ingest_text(text, args.source.name, client=client, tier=args.tier)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
