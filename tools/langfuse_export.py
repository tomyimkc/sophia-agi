#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ship sophia-contract trace spans to Langfuse.

    python tools/langfuse_export.py <traces.jsonl> [--dry-run]

Creds via env: LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY.
With --dry-run (or missing creds) it prints the batch it would send. Offline-safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_contract.langfuse_export import export_traces_file  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path, help="path to traces.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    result = export_traces_file(str(args.traces), dry_run=args.dry_run)
    if result.get("batch") is not None:
        print(json.dumps(result["batch"], indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "batch"}, indent=2))
    return 0 if result.get("sent") or args.dry_run or result.get("reason") else 1


if __name__ == "__main__":
    raise SystemExit(main())
