#!/usr/bin/env python3
"""The Sophia local-model evaluation ladder.

Compares the rungs the no-overclaim promotion rule needs:
    base · base+gate · adapter · adapter+gate · adapter+gate+MCP
recording each so "uplift" is always measured against a stored baseline, never vibes.

This runner verifies WIRING in --dry-run (no weights, CI-safe). The REAL rungs need
model weights on your hardware (Mac/MLX or a GPU box) — it prints those exact commands.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = "tools/eval_local_model.py"
DOMAINS = ["philosophy", "psychology", "history", "religion"]


def _rungs(model: str, adapter: str | None) -> list[tuple[str, list[str]]]:
    base = ["python", EVAL, "--model", model, "--domains", *DOMAINS]
    rungs = [("base", base), ("base+gate", base + ["--with-gate"])]
    if adapter:
        a = ["python", EVAL, "--adapter", adapter, "--domains", *DOMAINS]
        rungs += [("adapter", a), ("adapter+gate", a + ["--with-gate"])]
    return rungs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dry-run", action="store_true", help="verify wiring only (no weights)")
    args = ap.parse_args(argv)

    rungs = _rungs(args.model, args.adapter)
    print("Sophia eval ladder — promotion rule: improve provenance/citation at "
          "acceptable false-positive cost (no useful-correctness regression).\n")
    for name, cmd in rungs:
        shown = " ".join(cmd)
        if args.dry_run:
            rc = subprocess.run(cmd + ["--dry-run"], cwd=ROOT).returncode
            print(f"[{name}] wiring {'OK' if rc == 0 else 'FAIL'} :: {shown} --dry-run")
            if rc != 0:
                return 1
        else:
            print(f"[{name}] RUN ON YOUR HARDWARE: {shown}")
    print("\nAlso run on real weights: tools/run_seib.py --real-model --model <m>; "
          "run_all_phase_benchmarks.py; run_council_uplift.py; run_moral_public_standard_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
