#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fair-baseline LoRA speedup benchmark (feasibility experiment #2).

Measures the HONEST wall-clock multiplier of the optimized training stack against a
"standard LoRA" reference — SAME model, SAME data, SAME 1 epoch. It does NOT measure
the corpus-shrink lever (training on 500-2000 rows instead of tens of thousands); that
is a different model, not the same job done faster, and conflating the two is how the
"10-50x" claim overstates. This script isolates the apples-to-apples backend/padding
effect, which is realistically ~2-4x.

Configs (each a `tools/train_lora.py` invocation over the same subset, 1 epoch, eval
disabled so timing is pure train):

  peft-fp16-maxpad   standard-LoRA reference: vanilla PEFT, pad every batch to max-seq-len
  peft-fp16-dynpad   same, but dynamic per-batch padding (isolates the padding win)
  peft-4bit-dynpad   QLoRA 4-bit + dynamic padding
  unsloth-4bit-dynpad Unsloth fused kernels + 4-bit + dynamic padding

Real numbers need a CUDA GPU (RTX 4090 etc.). Without one each run exits fast with
"CUDA not detected" and is recorded as skipped. Use --dry-run to validate plumbing and
print the exact commands offline.

    python tools/bench_lora_speedup.py --dry-run
    python tools/bench_lora_speedup.py --limit 64 --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = ROOT / "training" / "lora" / "train.jsonl"

# label -> extra flags appended to the common train_lora invocation.
CONFIGS: list[dict] = [
    {"label": "peft-fp16-maxpad", "ref": True, "flags": ["--dtype", "fp16", "--pad-to-max"],
     "note": "standard-LoRA reference (full-length padding)"},
    {"label": "peft-fp16-dynpad", "flags": ["--dtype", "fp16"],
     "note": "dynamic per-batch padding (isolates the padding win)"},
    {"label": "peft-4bit-dynpad", "flags": ["--4bit"],
     "note": "QLoRA 4-bit + dynamic padding"},
    {"label": "unsloth-4bit-dynpad", "flags": ["--backend", "unsloth", "--4bit"],
     "note": "Unsloth fused kernels + 4-bit + dynamic padding"},
]


def _write_subset(src: Path, dst: Path, limit: int) -> int:
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit:
        lines = lines[:limit]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _build_cmd(cfg: dict, *, model: str, train: Path, output: Path, holdout: Path, seed: int) -> list[str]:
    return [
        sys.executable, str(ROOT / "tools" / "train_lora.py"),
        "--model", model,
        "--train", str(train),
        "--output", str(output),
        "--holdout", str(holdout),  # nonexistent -> eval disabled -> pure train timing
        "--epochs", "1",
        "--eval-every", "0",
        "--seed", str(seed),
        *cfg["flags"],
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--train", type=Path, default=TRAIN_JSONL)
    ap.add_argument("--limit", type=int, default=64, help="rows of the subset to time (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="comma-separated config labels to run (default: all)")
    ap.add_argument("--workdir", type=Path, default=ROOT / "training" / "lora" / "bench")
    ap.add_argument("--out", type=Path, default=ROOT / "training" / "lora" / "bench" / "speedup_report.json")
    ap.add_argument("--dry-run", action="store_true", help="print commands; do not run")
    args = ap.parse_args(argv)

    if not args.train.exists():
        print(f"Missing {args.train}. Run: python tools/prepare_lora_dataset.py")
        return 1

    args.workdir.mkdir(parents=True, exist_ok=True)
    subset = args.workdir / "subset.jsonl"
    n_rows = _write_subset(args.train, subset, args.limit)
    holdout = args.workdir / "_no_holdout.jsonl"  # intentionally absent on disk
    if holdout.exists():
        holdout.unlink()

    selected = set(s.strip() for s in args.only.split(",")) if args.only else None
    configs = [c for c in CONFIGS if not selected or c["label"] in selected]

    print(f"Bench: model={args.model} rows={n_rows} epoch=1 seed={args.seed} "
          f"configs={[c['label'] for c in configs]}\n", flush=True)

    results: list[dict] = []
    for cfg in configs:
        out_dir = args.workdir / cfg["label"]
        cmd = _build_cmd(cfg, model=args.model, train=subset, output=out_dir, holdout=holdout, seed=args.seed)
        print(f"[{cfg['label']}] {cfg.get('note', '')}\n  $ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            results.append({"label": cfg["label"], "ref": cfg.get("ref", False), "dryRun": True})
            continue

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        tail = proc.stdout[-2000:]
        skipped = "CUDA GPU not detected" in proc.stdout or "cuda available: False" in proc.stdout
        steps = None
        m = re.search(r"Run summary:\s*(\{.*\})", proc.stdout)
        if m:
            try:
                steps = json.loads(m.group(1)).get("globalSteps")
            except Exception:  # noqa: BLE001
                steps = None
        rec = {
            "label": cfg["label"], "ref": cfg.get("ref", False),
            "wallSeconds": round(wall, 2), "returnCode": proc.returncode,
            "skipped": skipped, "globalSteps": steps,
        }
        if proc.returncode != 0 and not skipped:
            rec["stderrTail"] = proc.stderr[-600:]
        results.append(rec)
        status = "SKIPPED (no GPU)" if skipped else (f"{wall:.1f}s" if proc.returncode == 0 else f"FAILED rc={proc.returncode}")
        print(f"  -> {status}\n", flush=True)

    # Speedup vs the standard-LoRA reference (only over runs that actually trained).
    ref = next((r for r in results if r.get("ref")), None)
    ref_wall = ref.get("wallSeconds") if ref and not ref.get("skipped") and ref.get("returnCode") == 0 else None
    for r in results:
        if ref_wall and not r.get("skipped") and r.get("returnCode") == 0 and r.get("wallSeconds"):
            r["speedupVsRef"] = round(ref_wall / r["wallSeconds"], 2)

    report = {
        "schema": "sophia.lora_speedup_bench.v1",
        "claimBoundary": "Apples-to-apples backend/padding speedup at FIXED data+epochs. Does NOT "
                         "measure the corpus-shrink lever; the headline 10-50x conflates the two.",
        "model": args.model, "rows": n_rows, "epochs": 1, "seed": args.seed,
        "results": results,
        "referenceLabel": ref["label"] if ref else None,
    }
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    print("\nlabel                  wall(s)   speedup_vs_ref")
    for r in results:
        if r.get("dryRun"):
            continue
        sp = r.get("speedupVsRef")
        wall = r.get("wallSeconds")
        print(f"  {r['label']:22}{('skip' if r.get('skipped') else wall):>7}   {sp if sp else '-'}")
    if not args.dry_run and ref_wall is None:
        print("\nNote: reference run did not train (no GPU?), so speedups are unavailable. "
              "Run on a CUDA GPU for real numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
