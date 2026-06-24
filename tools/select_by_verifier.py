#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-driven data selection: concentrate training on the boundary cases.

The gate is the verifier. Given an existing SFT JSONL and a model client, for each
prompt we sample ONE answer from the model and run the INTRINSIC fail-closed gate
on it -- ``check_response(text, mode="advisor")["violations"]`` WITHOUT a question
(no trap grader; mirrors tools/train_lora.py:guard_filter and tools/run_rft.py).

  * If the model's sampled answer is gate-CLEAN, the model ALREADY passes this
    prompt -> DROP the row (no marginal training value).
  * If the model's sampled answer is gate-FLAGGED (a violation), the model
    currently FAILS this prompt -> KEEP the row (a boundary case worth training).

This is the verifier-gated analogue of "select the examples the model gets wrong":
training compute is spent where the model is still wrong, not re-teaching what it
already does. The KEPT rows are the original curated targets, untouched -- we only
use the model's own sampled answer as the difficulty signal, never as the target,
so curated discipline (including correct abstentions) is preserved verbatim.

Fail-closed note: a row whose model-sample is empty/failed is conservatively KEPT
(treated as "currently fails") so a flaky client never silently discards data.

    # offline plumbing (mock client, writes nothing)
    python tools/select_by_verifier.py --in training/lora/train.jsonl --dry-run
    # real selection pass
    python tools/select_by_verifier.py --in training/lora/train.jsonl \
        --out training/lora/boundary.jsonl --model openrouter:deepseek/deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402

DEFAULT_IN = ROOT / "training" / "lora" / "train.jsonl"
DEFAULT_OUT = ROOT / "training" / "lora" / "boundary.jsonl"

ADVISOR_SYSTEM = (
    "You are a source-disciplined advisor. Answer the question, cite a source where one is "
    "relied on, and ABSTAIN rather than guess if you cannot verify a needed authority or "
    "figure. Not professional advice; end with a 中文摘要."
)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _row_prompt(row: dict) -> "tuple[str, str]":
    """Return (system, user) for a {messages:[...]} row: last user turn as the
    question, any system turn (falling back to the advisor scaffold) as system."""
    msgs = row.get("messages") or []
    system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "") or ADVISOR_SYSTEM
    user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    return system, user


def _gen(client, system: str, user: str) -> str:
    try:
        res = client.generate(system, user)
    except Exception:  # noqa: BLE001 - a broken client yields no content, not a crash
        return ""
    return (getattr(res, "text", "") or "").strip() if getattr(res, "ok", False) else ""


def select(rows: list[dict], client) -> "tuple[list[dict], dict]":
    """Keep rows the model currently FAILS (gate-flagged sample), drop ones it
    already passes (gate-clean sample). Returns (kept_rows, stats)."""
    kept: list[dict] = []
    dropped_pass = kept_fail = kept_noprompt = 0
    for row in rows:
        system, user = _row_prompt(row)
        if not user.strip():
            kept.append(row)  # can't probe -> keep conservatively
            kept_noprompt += 1
            continue
        sample = _gen(client, system, user)
        # INTRINSIC gate: NO question -> fail-closed intrinsic check, no trap grader.
        # Empty/failed sample has no violations to find; treat it as "currently fails".
        flagged = (not sample.strip()) or bool(check_response(sample, mode="advisor")["violations"])
        if flagged:
            kept.append(row)
            kept_fail += 1
        else:
            dropped_pass += 1
    stats = {"in": len(rows), "kept": len(kept), "dropped": dropped_pass,
             "keptModelFails": kept_fail, "keptNoPrompt": kept_noprompt,
             "droppedModelPasses": dropped_pass}
    return kept, stats


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", default=str(DEFAULT_IN), help="input SFT JSONL")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output JSONL of kept boundary rows")
    ap.add_argument("--model", default="mock", help="model spec (default mock = offline plumbing)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + row count; write nothing")
    args = ap.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    if not in_path.exists():
        print(f"Missing input {in_path}", flush=True)
        return 1
    rows = load_rows(in_path)

    if args.dry_run:
        plan = {
            "model": args.model,
            "in": str(in_path),
            "rows": len(rows),
            "rule": "DROP prompts the model already passes (gate-clean sample); "
                    "KEEP prompts it currently fails (gate-flagged) — boundary cases",
            "gate": "intrinsic (mode=advisor, no question) — fail-closed",
            "out": args.out,
        }
        print("Verifier-selection plan (dry-run, nothing written):", flush=True)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0

    from agent.model import default_client
    kept, stats = select(rows, default_client(args.model))
    stats["model"] = args.model

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out  # output path outside the repo (e.g. /tmp) — show it absolute
    print(f"kept {len(kept)} boundary row(s), dropped {stats['dropped']} -> {shown}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
