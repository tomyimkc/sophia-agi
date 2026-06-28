#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Stamp a JSONL training pack with per-row data passports + emit a datasheet.

    python -m pretraining.data_passport.build_passport <pack.jsonl> [--out stamped.jsonl] \
        [--datasheet datasheet.json]

Reads SFT (messages) or DPO/plain rows, attaches a ``_passport`` to each, and prints a
datasheet (dup rate, license/source breakdown, quality, flags). Also runs the existing
contamination guard if available, so the datasheet records eval-leak status too.
Pure stdlib, offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.data_passport.passport import stamp_pack


def _load_jsonl(path: Path) -> "list[dict]":
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _contamination(rows: "list[dict]") -> dict | None:
    """Best-effort hook into the existing contamination guard (optional)."""
    try:
        from provenance_bench.dataset_guard import check_contamination
        return check_contamination(rows)
    except Exception as exc:  # noqa: BLE001 - guard is optional in some checkouts
        return {"available": False, "reason": str(exc)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pack", type=Path, help="JSONL training pack")
    ap.add_argument("--out", type=Path, default=None, help="write stamped JSONL here")
    ap.add_argument("--datasheet", type=Path, default=None, help="write datasheet JSON here")
    ap.add_argument("--near-dup", type=float, default=0.8)
    ap.add_argument("--quality-floor", type=float, default=0.35)
    args = ap.parse_args()

    rows = _load_jsonl(args.pack)
    result = stamp_pack(rows, near_dup_threshold=args.near_dup,
                        quality_floor=args.quality_floor)
    datasheet = result["datasheet"]
    datasheet["pack"] = str(args.pack)
    datasheet["contamination"] = _contamination(rows)

    if args.out:
        with args.out.open("w", encoding="utf-8") as fh:
            for r in result["rows"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.datasheet:
        args.datasheet.write_text(json.dumps(datasheet, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    print(json.dumps(datasheet, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
