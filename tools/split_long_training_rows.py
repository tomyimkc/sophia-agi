#!/usr/bin/env python3
"""Fit MLX chat-training rows under a token budget so nothing is silently truncated.

The v2 MLX run (`local-sophia-v2-mlx-trained-not-promoted-2026-06-24`) emitted
"prompt exceeds max_seq_length, truncating" warnings: rows longer than `--max-seq-length`
were cut mid-example, corrupting supervision without any record. This module guarantees
**every emitted row fits**, by either keeping it, splitting a multi-turn conversation at
turn boundaries into fitting sub-rows, or dropping it — and it records the counts.

Token counting is, by default, an offline heuristic (chars-per-token + per-message
overhead) so this runs in CI with no model download. Pass `--hf-tokenizer <name-or-path>`
to use the real tokenizer when `transformers` is installed (the heuristic is intentionally
*conservative* — it slightly over-counts — so the offline default never lets a too-long row
through).

This changes no weights and is not an AGI claim; it is training-data hygiene.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]

# Conservative offline defaults. ~3.5 chars/token under-estimates length → over-estimates
# tokens for English+code, which is the safe direction (we'd rather split/drop a borderline
# row than let it truncate).
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CHARS_PER_TOKEN = 3.5
DEFAULT_PER_MESSAGE_OVERHEAD = 4

TokenCounter = Callable[[list[dict]], int]


def heuristic_counter(
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    per_message_overhead: int = DEFAULT_PER_MESSAGE_OVERHEAD,
) -> TokenCounter:
    def count(messages: list[dict]) -> int:
        total = 0
        for m in messages:
            content = str(m.get("content", ""))
            total += per_message_overhead + math.ceil(len(content) / chars_per_token)
        return total
    return count


def hf_counter(name_or_path: str) -> TokenCounter:
    """Real-tokenizer counter; only importable where transformers is installed."""
    from transformers import AutoTokenizer  # lazy: absent in CI/sandbox

    tok = AutoTokenizer.from_pretrained(name_or_path)

    def count(messages: list[dict]) -> int:
        try:
            ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            return len(ids)
        except Exception:
            text = "\n".join(str(m.get("content", "")) for m in messages)
            return len(tok.encode(text))
    return count


def _leading_system(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    i = 0
    while i < len(messages) and str(messages[i].get("role")) == "system":
        i += 1
    return messages[:i], messages[i:]


def _turn_pairs(messages: list[dict]) -> list[list[dict]]:
    """Group the non-system tail into [user, assistant, ...] chunks at each user turn."""
    pairs: list[list[dict]] = []
    current: list[dict] = []
    for m in messages:
        if str(m.get("role")) == "user" and current:
            pairs.append(current)
            current = [m]
        else:
            current.append(m)
    if current:
        pairs.append(current)
    return pairs


def fit_row(row: dict, *, max_tokens: int, count: TokenCounter) -> tuple[list[dict], dict[str, int]]:
    """Return (sub-rows that each fit, stats). Greedily packs turn-pairs to preserve
    multi-turn context; drops any single turn that cannot fit even alone."""
    stats = {"kept": 0, "split": 0, "droppedUnsplittableTurn": 0}
    messages = row.get("messages") or []
    if count(messages) <= max_tokens:
        stats["kept"] = 1
        return [row], stats

    system, tail = _leading_system(messages)
    pairs = _turn_pairs(tail)
    out: list[dict] = []
    current = list(system)
    meta = row.get("metadata", {})

    def emit(msgs: list[dict]) -> None:
        if len(msgs) > len(system):  # has real content beyond the system preamble
            out.append({"messages": msgs, "metadata": {**meta, "split": True}})

    for pair in pairs:
        candidate = current + pair
        if count(candidate) <= max_tokens:
            current = candidate
            continue
        emit(current)
        if count(system + pair) <= max_tokens:
            current = system + pair
        else:
            stats["droppedUnsplittableTurn"] += 1
            current = list(system)
    emit(current)

    if not out:
        return [], stats  # whole row unsplittable
    stats["split"] = len(out)
    return out, stats


def fit_rows(rows: list[dict], *, max_tokens: int = DEFAULT_MAX_TOKENS, count: TokenCounter | None = None) -> tuple[list[dict], dict[str, Any]]:
    counter = count or heuristic_counter()
    kept: list[dict] = []
    report = {
        "maxTokens": max_tokens,
        "rowsIn": len(rows),
        "rowsKeptUnchanged": 0,
        "rowsSplit": 0,
        "subRowsFromSplits": 0,
        "rowsDroppedUnsplittable": 0,
        "turnsDroppedOverlong": 0,
        "rowsOut": 0,
    }
    for row in rows:
        subs, stats = fit_row(row, max_tokens=max_tokens, count=counter)
        report["turnsDroppedOverlong"] += stats["droppedUnsplittableTurn"]
        if stats["kept"]:
            report["rowsKeptUnchanged"] += 1
            kept.extend(subs)
        elif subs:
            report["rowsSplit"] += 1
            report["subRowsFromSplits"] += len(subs)
            kept.extend(subs)
        else:
            report["rowsDroppedUnsplittable"] += 1
    report["rowsOut"] = len(kept)
    # Invariant: every emitted row fits. (Cheap to assert; this is the whole point.)
    assert all(counter(r.get("messages") or []) <= max_tokens for r in kept), "fit_rows left an overlong row"
    return kept, report


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="MLX chat JSONL file(s) to fit in place")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--hf-tokenizer", default=None, help="use this HF tokenizer (needs transformers)")
    ap.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not rewrite files")
    args = ap.parse_args(argv)

    counter = hf_counter(args.hf_tokenizer) if args.hf_tokenizer else heuristic_counter(args.chars_per_token)
    rc = 0
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            print(f"skip (missing): {raw}")
            continue
        rows = _read_jsonl(p)
        fitted, report = fit_rows(rows, max_tokens=args.max_tokens, count=counter)
        report["tokenizer"] = args.hf_tokenizer or f"heuristic(chars_per_token={args.chars_per_token})"
        print(json.dumps({"file": raw, **report}, indent=2))
        if not args.dry_run:
            _write_jsonl(p, fitted)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
