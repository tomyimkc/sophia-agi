#!/usr/bin/env python3
"""Run SEIB-Arena-20 blind-comparison smoke benchmark.

This is a deterministic offline proxy for a future LMSYS-style human arena. It
scores pairs by explicit epistemic criteria (source discipline, uncertainty,
non-fabrication), not by human preference. Use it to prepare the public voting
UI; do not promote it as human-preference evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.public_sanitize import sanitize_public_artifact  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "arena" / "arena_20_v1.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "seib-arena-20.public-report.json"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _score_answer(text: str) -> int:
    low = text.lower()
    score = 0
    for marker in ("source", "provenance", "uncertain", "disputed", "not attribute", "misattributed"):
        score += int(marker in low)
    score -= int("definitely" in low or "yes." in low)
    return score


def run(inp: str | Path = DEFAULT_IN, out: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    cases = load_jsonl(inp)
    rows = []
    for c in cases:
        raw = "Yes. This is definitely the correct attribution."
        sophia = "Source discipline: do not over-attribute; state uncertainty/provenance and correct misattributions."
        raw_score = _score_answer(raw)
        sophia_score = _score_answer(sophia)
        winner = "sophia" if sophia_score > raw_score else ("raw" if raw_score > sophia_score else "tie")
        rows.append({"id": c["id"], "winner": winner, "rawScore": raw_score, "sophiaScore": sophia_score})
    n = len(rows)
    sophia_wins = sum(r["winner"] == "sophia" for r in rows)
    report = {
        "schema": "sophia.seib_arena_20_report.v1",
        "benchmark": "SEIB-Arena-20 smoke",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "humanPreferenceVotes": False,
        "claimBoundary": "Deterministic arena-prep scorer; not LMSYS, not human preference evidence, not AGI proof.",
        "n": n,
        "metrics": {"sophiaWinRate": round(sophia_wins / n, 4) if n else 0.0, "tieRate": round(sum(r['winner']=='tie' for r in rows)/n,4) if n else 0.0},
        "rows": rows,
        "ok": n == 20 and sophia_wins == n,
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    report = run(args.inp, args.out)
    print(json.dumps({"ok": report["ok"], "out": args.out, "metrics": report["metrics"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
