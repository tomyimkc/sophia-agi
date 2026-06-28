#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Materialize MLX-LM chat data for the math-code curriculum pack.

Copies oracle-verified rows into ``training/sophia-math-code-curriculum/mlx/train.jsonl``
without duplicating the pack in git (run locally before ``train_lora.py --backend mlx``).

Usage:
  python tools/prepare_math_code_mlx.py
  python tools/prepare_math_code_mlx.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACK = ROOT / "training" / "sophia-math-code-curriculum"
SRC = PACK / "sft_all.jsonl"
OUT_DIR = PACK / "mlx"
OUT = OUT_DIR / "train.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"Missing {SRC}. Run: python tools/generate_math_code_curriculum.py", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            msgs = row.get("messages")
            if isinstance(msgs, list) and msgs:
                rows.append({"messages": msgs, "metadata": row.get("metadata", {})})

    print(json.dumps({"source": str(SRC.relative_to(ROOT)), "rows": len(rows), "out": str(OUT.relative_to(ROOT))}))
    if args.dry_run:
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
