#!/usr/bin/env python3
"""Run the Continual Provenance QA (CPQA) benchmark.

Streams a sequence of episodes (learn / retract / query) through two systems and scores
the integrated dual-store loop against a frozen weight-model baseline:

- graph_backed: knowledge in the OKF belief graph; learns by page write, revises
  conflicts, unlearns on demand, answers only from the grounded belief state (Experiments
  1–4 integrated).
- parametric_baseline: knowledge frozen after episode 0 — cannot learn new facts, cannot
  unlearn/correct stale ones.

Deterministic, offline. Writes a public report to agi-proof/benchmark-results/.

    python tools/run_continual_qa_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa import load_episodes, run_benchmark  # noqa: E402
from agent.public_sanitize import sanitize_public_artifact  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "continual_qa" / "episodes_v1.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "continual-qa.public-report.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Continual Provenance QA benchmark.")
    ap.add_argument("--episodes", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run_benchmark(load_episodes(args.episodes))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")

    gb = report["systems"]["graph_backed"]
    bl = report["systems"]["parametric_baseline"]
    print(json.dumps({
        "out": args.out,
        "queries": report["queryCount"],
        "graph_backed": {"accuracy": gb["accuracy"], "fabricationRate": gb["fabricationRate"],
                         "missRate": gb["missRate"]},
        "parametric_baseline": {"accuracy": bl["accuracy"], "fabricationRate": bl["fabricationRate"],
                                "missRate": bl["missRate"]},
        "unintendedForgetting": report["retention"]["unintendedForgetting"],
        "deliberateUnlearning": report["retention"]["deliberateUnlearning"],
    }, indent=2))


if __name__ == "__main__":
    main()
