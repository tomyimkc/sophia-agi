#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build + validate per-discipline distillation seed packs.

``training/council_seeds/<discipline>.jsonl`` holds hand-authored teacher CoT traces — the
gate-clean nucleus each ``sophia-<discipline>-3b`` adapter is SFT-seeded on (Stage 1 of the 3B plan).
This runs every seed through the **discipline-aware** distiller (``tools/gen_reasoning_distill``,
which gates each answer by ITS seat's verifier) and:

  * asserts EVERY seed is gate-clean (a dropped seed is a SEED BUG to fix, not training noise);
  * emits the combined ``<think>``-delimited SFT pack ``training/council_seeds/distill_v1.jsonl``;
  * reports kept/dropped per discipline.

So the seed corpus is, by construction, only reasoning whose answer its own discipline verifier
accepts. Real volume is added later from a teacher model through the same gate. Deterministic,
offline. canClaimAGI stays false.

    python tools/build_council_seeds.py            # validate + emit distill_v1.jsonl
    python tools/build_council_seeds.py --check    # validate only, exit 1 if any seed drops
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gen_reasoning_distill import row_from_trace  # noqa: E402

SEEDS_DIR = ROOT / "training" / "council_seeds"
OUT = SEEDS_DIR / "distill_v1.jsonl"


def _read(path: Path) -> "list[dict]":
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build() -> "tuple[list[dict], dict]":
    out: list[dict] = []
    per_discipline: dict = {}
    drops: list[dict] = []
    for path in sorted(SEEDS_DIR.glob("*.jsonl")):
        if path.name == OUT.name:
            continue
        for trace in _read(path):
            disc = trace.get("discipline") or path.stem
            pd = per_discipline.setdefault(disc, {"total": 0, "kept": 0, "dropped": 0})
            pd["total"] += 1
            row, reason = row_from_trace(trace)
            if row:
                pd["kept"] += 1
                out.append(row)
            else:
                pd["dropped"] += 1
                drops.append({"discipline": disc, "caseId": trace.get("caseId"), "reason": reason})
    return out, {"perDiscipline": per_discipline, "drops": drops,
                 "total": sum(p["total"] for p in per_discipline.values()),
                 "kept": len(out)}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate only; exit 1 if any seed drops")
    args = ap.parse_args(argv)

    rows, stats = build()
    clean = len(stats["drops"]) == 0
    print(json.dumps({k: v for k, v in stats.items() if k != "drops"}, ensure_ascii=False, indent=2))
    if stats["drops"]:
        print("SEED BUGS (answer failed its discipline gate — fix the seed):")
        print(json.dumps(stats["drops"], ensure_ascii=False, indent=2))
    if not args.check:
        OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"wrote {len(rows)} SFT rows -> {OUT.relative_to(ROOT)}")
    print("SEEDS:", "ALL GATE-CLEAN" if clean else "DROPS PRESENT")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
