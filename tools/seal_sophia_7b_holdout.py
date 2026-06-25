#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seal the LoRA holdout for the sophia-7b-train-verify pre-registration.

Writes a prompt-only manifest (no assistant gold) and verifies it on --check.

    python tools/seal_sophia_7b_holdout.py
    python tools/seal_sophia_7b_holdout.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.holdout_seal import (  # noqa: E402
    DEFAULT_HOLDOUT,
    DEFAULT_MANIFEST,
    build_manifest,
    verify_manifest,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--check", action="store_true", help="verify manifest matches holdout")
    args = ap.parse_args(argv)

    if args.check:
        if not args.out.exists():
            print(f"::error:: manifest missing: {args.out}", file=sys.stderr)
            return 1
        result = verify_manifest(args.out, args.holdout)
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            print("::error:: holdout seal mismatch — manifest stale or holdout tampered", file=sys.stderr)
            return 1
        print("holdout seal: OK")
        return 0

    manifest = build_manifest(args.holdout, base_model=args.base_model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"contentHash: {manifest['contentHash']}")
    print(f"rowCount: {manifest['rowCount']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
