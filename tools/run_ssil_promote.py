#!/usr/bin/env python3
"""Live MEASURED SSIL loop — the path where a real improvement can PROMOTE.

The model proposes an executable routing policy (thresholds) from the TRAIN feature
summary only; we measure its real held-out TEST gain vs baseline, probe corrigibility
and honeypots, and run all gates. A promotion requires a *measured* gain with no
protected regression — and the model still never sees the test labels or its own score.

Requires a provider key in env (e.g. DEEPSEEK_API_KEY); falls back to offline mock.
NEVER hardcode a key. Output carries candidateOnly / canClaimAGI=false.

  export DEEPSEEK_API_KEY=...   # rotate after use
  python3 tools/run_ssil_promote.py

See docs/11-Platform/Safe-Self-Improvement-Loop.md (G4 + live proposer).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_proposer import run_microtask_ssil  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-extension" / "ssil-promote.public-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Live measured SSIL loop (can promote)")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    rec = run_microtask_ssil(spec=args.spec, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    m = rec["measurement"]
    if args.print:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    print(f"provider={rec['provider']} policy={rec['proposedPolicy']}")
    print(f"baseline_acc={m['baseline']['accuracy']} candidate_acc={m['candidate']['accuracy']} "
          f"protected_recall={m['candidate']['answer_recall']}")
    print(f"VERDICT={rec['verdict']} blocking={rec['blockingGates']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
