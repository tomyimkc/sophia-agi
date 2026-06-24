#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the SSIL loop with a LIVE model proposer + behavioral probes.

The model proposes a self-modification and is probed on the frozen corrigibility
scenarios and active honeypots; the deterministic gates grade its actual behavior.
The model never declares its own verdicts and never sees its own score.

Requires a provider key in the environment (e.g. DEEPSEEK_API_KEY). NEVER pass a key
on the command line in a shared shell — export it. Falls back to the offline `mock`
provider if no key is set. Output: candidate infrastructure only
(``candidateOnly``/``canClaimAGI=false`` carried by the SSIL record).

  export DEEPSEEK_API_KEY=...   # rotate after use
  python3 tools/run_ssil_live.py "improve source-provenance routing"

See docs/11-Platform/Safe-Self-Improvement-Loop.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_proposer import run_live_ssil  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-extension" / "ssil-live.public-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="Live SSIL loop (model proposer + probes)")
    ap.add_argument("task", nargs="?", default="improve source-provenance routing")
    ap.add_argument("--spec", default=None, help="model spec (default: deepseek if key set, else mock)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    record = run_live_ssil(args.task, spec=args.spec, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"provider={record['provider']} candidate={record['candidateId']} verdict={record['verdict']} blocking={record['blockingGates']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
