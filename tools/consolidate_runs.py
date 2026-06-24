#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Consolidate agent run logs into gated OKF memory pages (episodic -> semantic).

Folds every verified run in agent/memory/agent_runs/*.jsonl into the wiki memory
tier, benchmark-leak guarded and provenance-gated, so the agent's knowledge
compounds across runs without retraining. Answers that merged a lineage are
rejected, never consolidated.

    python tools/consolidate_runs.py            # fold all runs into agent/memory/wiki/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.memory_consolidation import RUNS_DIR, consolidate_runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolidate agent runs into OKF memory pages")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--no-deleak", action="store_true", help="skip benchmark leakage check")
    parser.add_argument("--tier", default="memory", choices=["memory", "draft"])
    args = parser.parse_args()
    result = consolidate_runs(args.runs_dir, deleak=not args.no_deleak, tier=args.tier)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
