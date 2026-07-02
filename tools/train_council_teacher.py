#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A3 — train a REAL council-seat teacher via the Agents-A1 two-stage recipe.

Closes the named open gap "real trained discipline adapters" (council seats in
agent/council_registry.py are prompt/verifier stubs today) using the cheapest
teacher recipe in Agents-A1 (arXiv 2606.30616 §4.2.2, their largest single
delta, no RL required):

  Stage 1 — reasoning-enhanced SFT: pure reasoning traces for the seat.
  Stage 2 — tool-augmented SFT CONTINUED from the stage-1 adapter, with
            extended iterations (stabilizes tool use while retaining the
            stage-1 reasoning style).

Guardrails (do not bypass):
  * PROTECTED seats (history, religion) are refused outright — the registry
    marks them never-RL-optimised and this tool honors that for SFT teachers
    too: a teacher that drifts a protected domain is exactly the regression
    the promotion gate exists to reject, so we do not create the risk.
  * The trained adapter is a CANDIDATE: acceptance goes through
    agent/adapter_registry.py (>=2 judge families, CI excluding 0, kappa>=0.40)
    and tools/promote_adapter.py — never through this tool's exit code.
  * Fail-closed: no MLX backend -> plan-only; missing/empty data -> refusal.

Usage:
  PYTHONPATH=. python3 tools/train_council_teacher.py --seat philosophy \
      --stage1-data training/teachers/philosophy/stage1/train.jsonl \
      --stage2-data training/teachers/philosophy/stage2/train.jsonl --plan  # validate only
  (--stage*-data point at the train.jsonl INSIDE each stage dir built by
   tools/build_teacher_data.py; the stage DIRECTORY is what mlx_lm consumes)
  (drop --plan on the Mac bench to actually train; runs mlx_lm lora twice)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sophia.council_teacher_training.v1"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"  # the repo's frozen base (chart 8)


def build_plan(seat: str, stage1_data: Path, stage2_data: Path, *,
               adapter_out: "Path | None" = None, iters1: int = 300,
               iters2: int = 500, batch_size: int = 4,
               max_seq_len: int = 1024) -> "dict[str, Any]":
    """Validate inputs and produce the two-stage training plan (fail-closed)."""
    from agent.council_registry import DISCIPLINES

    if seat not in DISCIPLINES:
        return {"schema": SCHEMA, "ok": False, "candidateOnly": True,
                "reason": f"unknown council seat {seat!r}; known: {sorted(DISCIPLINES)}"}
    disc = DISCIPLINES[seat]
    if disc.protected:
        return {"schema": SCHEMA, "ok": False, "candidateOnly": True,
                "reason": f"seat {seat!r} is PROTECTED (never tuned); refusing fail-closed"}
    for label, p in (("stage1", stage1_data), ("stage2", stage2_data)):
        if not p.exists() or not p.read_text(encoding="utf-8").strip():
            return {"schema": SCHEMA, "ok": False, "candidateOnly": True,
                    "reason": f"{label} data missing/empty: {p} (fail-closed; author real "
                              "traces, e.g. via tools/build_discipline_sft.py or the A1 pack)"}
    out = adapter_out or (ROOT / "training" / "mlx_adapters" / str(disc.adapter_slot))
    stage1_dir = out.parent / (out.name + "-stage1")
    common = ["--model", BASE_MODEL, "--batch-size", str(batch_size),
              "--mask-prompt", "--max-seq-length", str(max_seq_len)]
    return {
        "schema": SCHEMA, "ok": True, "candidateOnly": True, "level3Evidence": False,
        "seat": seat, "adapterSlot": disc.adapter_slot, "baseModel": BASE_MODEL,
        "stages": [
            {"name": "stage1-reasoning-sft",
             "argv": ["python3", "-m", "mlx_lm", "lora", "--train",
                      "--data", str(stage1_data.parent), "--iters", str(iters1),
                      "--adapter-path", str(stage1_dir), *common]},
            {"name": "stage2-tool-sft-continued",
             # extended iterations, resumed FROM the stage-1 adapter (the recipe's core)
             "argv": ["python3", "-m", "mlx_lm", "lora", "--train",
                      "--data", str(stage2_data.parent), "--iters", str(iters2),
                      "--resume-adapter-file", str(stage1_dir / "adapters.safetensors"),
                      "--adapter-path", str(out), *common]},
        ],
        "acceptance": "agent/adapter_registry.py AcceptanceEvidence (>=2 judge families, "
                      "CI excluding 0, kappa>=0.40) then tools/promote_adapter.py; this "
                      "tool trains a CANDIDATE only",
        "recipe": "Agents-A1 §4.2.2 two-stage specialist SFT (reasoning-first, then "
                  "tool-augmented continued, extended rounds)",
    }


def mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
        return True
    except Exception:
        return False


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="A3 two-stage council-teacher trainer")
    ap.add_argument("--seat", required=True)
    ap.add_argument("--stage1-data", type=Path, required=True)
    ap.add_argument("--stage2-data", type=Path, required=True)
    ap.add_argument("--adapter-out", type=Path, default=None)
    ap.add_argument("--iters1", type=int, default=300)
    ap.add_argument("--iters2", type=int, default=500)
    ap.add_argument("--plan", action="store_true", help="validate + print plan, do not train")
    args = ap.parse_args(argv)

    plan = build_plan(args.seat, args.stage1_data, args.stage2_data,
                      adapter_out=args.adapter_out, iters1=args.iters1, iters2=args.iters2)
    print(json.dumps(plan, indent=2))
    if not plan["ok"]:
        return 2
    if args.plan:
        return 0
    if not mlx_available():
        print("MLX backend unavailable; refusing to train (run on the Mac bench). "
              "Plan above is valid — dispatch mac-mlx-bench or run locally on Apple "
              "Silicon.", file=sys.stderr)
        return 3
    for stage in plan["stages"]:
        print(f"[stage] {stage['name']}", flush=True)
        proc = subprocess.run(stage["argv"], cwd=ROOT, check=False)
        if proc.returncode != 0:
            print(f"stage {stage['name']} failed rc={proc.returncode}; aborting fail-closed",
                  file=sys.stderr)
            return proc.returncode
    print(json.dumps({"schema": SCHEMA, "ok": True, "trained": plan["adapterSlot"],
                      "candidateOnly": True,
                      "next": "eval ladder + adapter_registry acceptance + promote_adapter"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
