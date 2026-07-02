#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A3 data prep — build the two-stage teacher pack for a council seat.

Produces the directory layout ``tools/train_council_teacher.py`` (and
``mlx_lm lora --data``) consumes:

    training/teachers/<seat>/stage1/{train,valid}.jsonl   # reasoning traces
    training/teachers/<seat>/stage2/{train,valid}.jsonl   # tool-augmented traces

Sources are explicit (``--stage1-from`` / ``--stage2-from``, repeatable) — the
seat owner chooses them. Deliberately NOT routed through council_registry.route:
the v1 lexical router keys on words like "compile(d)", which routes
"who COMPILED the Analects" to the coding seat — a data-poisoning footgun for
teacher packs (verified 2026-07-02).

Guardrails: PROTECTED seats refused; every row decontaminated against
eval/holdout with the same guard as the main dataset builder; deterministic
90/10 split under --seed; fail-closed below --min-rows. candidateOnly: a data
pack claims nothing.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sophia.teacher_data_pack.v1"


def _load_rows(paths: "Sequence[Path]") -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if isinstance(r.get("messages"), list) and r["messages"]:
                rows.append({"messages": r["messages"], "metadata": r.get("metadata", {})})
    return rows


def _decontaminate(rows: "list[dict]", root: Path) -> "tuple[list[dict], int]":
    from provenance_bench.dataset_guard import eval_prompt_set, normalize, prompt_of

    forbidden = eval_prompt_set(root=root)
    kept = [r for r in rows
            if not (prompt_of(r) and normalize(prompt_of(r)) in forbidden)]
    return kept, len(rows) - len(kept)


def build_stage(rows: "list[dict]", out_dir: Path, *, seed: int,
                valid_frac: float = 0.1, min_rows: int = 20) -> "dict[str, Any]":
    if len(rows) < min_rows:
        return {"ok": False, "reason": f"only {len(rows)} rows (< {min_rows}); fail-closed — "
                                       "author more traces before training a teacher on this"}
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    k = max(1, int(len(shuffled) * valid_frac))
    valid, train = shuffled[:k], shuffled[k:]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"ok": True, "train": len(train), "valid": len(valid)}


def build_teacher_pack(seat: str, stage1_from: "Sequence[Path]",
                       stage2_from: "Sequence[Path]", *, out_root: Path,
                       root: Path, seed: int = 0, min_rows: int = 20) -> "dict[str, Any]":
    from agent.council_registry import DISCIPLINES

    if seat not in DISCIPLINES:
        return {"schema": SCHEMA, "ok": False, "candidateOnly": True,
                "reason": f"unknown seat {seat!r}"}
    if DISCIPLINES[seat].protected:
        return {"schema": SCHEMA, "ok": False, "candidateOnly": True,
                "reason": f"seat {seat!r} is PROTECTED; refusing fail-closed"}
    report: dict[str, Any] = {"schema": SCHEMA, "ok": True, "seat": seat,
                              "candidateOnly": True, "level3Evidence": False,
                              "seed": seed, "stages": {}}
    for label, sources in (("stage1", stage1_from), ("stage2", stage2_from)):
        rows = _load_rows(sources)
        rows, dropped = _decontaminate(rows, root)
        stage = build_stage(rows, out_root / seat / label, seed=seed, min_rows=min_rows)
        stage["droppedForDecontamination"] = dropped
        stage["sources"] = [str(s) for s in sources]
        report["stages"][label] = stage
        if not stage["ok"]:
            report["ok"] = False
    return report


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="A3 teacher data pack builder")
    ap.add_argument("--seat", required=True)
    ap.add_argument("--stage1-from", type=Path, action="append", required=True)
    ap.add_argument("--stage2-from", type=Path, action="append", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("training/teachers"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rows", type=int, default=20)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    report = build_teacher_pack(args.seat, args.stage1_from, args.stage2_from,
                                out_root=args.out_root, root=root,
                                seed=args.seed, min_rows=args.min_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
