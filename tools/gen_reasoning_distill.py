#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated reasoning distillation — only gate-clean teacher traces become SFT data.

s1 (Muennighoff et al. 2025) made a strong reasoner by SFT on ~1k distilled chain-of-thought
traces. The risk for a provenance-native model: distilling a teacher's reasoning that REACHES A
WRONG OR HALLUCINATED ANSWER teaches the student to reason confidently toward an error — the exact
failure Sophia exists to stop.

This tool gates distillation by the machine verifier: a teacher trace becomes a ``<think>``-delimited
SFT row ONLY if its final answer clears ``agent.gate.check_response``. A trace whose answer carries a
hard violation (bad attribution / citation / arithmetic) is DROPPED (fail-closed), with a reason —
never distilled. The verifier is the labeller, not an LLM judge (same property as
``tools/gen_verifier_dpo.py``).

Input rows (JSONL)::

    {"prompt": "...", "question": "<optional>", "thinking": "<teacher CoT>", "answer": "...",
     "mode": "advisor|repo|life", "caseId": "..."}

Output rows (JSONL) — reasoning-model SFT format with explicit think tags::

    {"messages": [{"role": "user", "content": prompt},
                  {"role": "assistant", "content": "<think>\n{thinking}\n</think>\n{answer}"}],
     "metadata": {"verified": true, "label_source": "machine_verified", "verifier": "agent.gate"}}

Usage::

    python tools/gen_reasoning_distill.py --in traces.jsonl --out training/reasoning/distill.jsonl
    python tools/gen_reasoning_distill.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def _verify_answer(answer: str, *, question: str, mode: str) -> "list[str]":
    """Return the hard violations on the answer (empty == gate-clean)."""
    r = check_response(answer, mode=mode, question=question, route_claims=True)
    return list(r.get("violations") or [])


def row_from_trace(trace: dict) -> "tuple[dict | None, str | None]":
    """Build an SFT row from one teacher trace, or ``(None, reason)`` if it is dropped."""
    prompt = (trace.get("prompt") or "").strip()
    thinking = (trace.get("thinking") or "").strip()
    answer = (trace.get("answer") or "").strip()
    if not prompt or not answer:
        return None, "missing_prompt_or_answer"
    if not thinking:
        return None, "missing_thinking"

    question = (trace.get("question") or prompt).strip()
    mode = trace.get("mode") or "advisor"
    violations = _verify_answer(answer, question=question, mode=mode)
    if violations:
        return None, f"answer_failed_gate:{violations[:2]}"

    content = f"{THINK_OPEN}\n{thinking}\n{THINK_CLOSE}\n{answer}"
    meta = {"verified": True, "label_source": "machine_verified", "verifier": "agent.gate"}
    if trace.get("caseId"):
        meta["caseId"] = trace["caseId"]
    row = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content},
        ],
        "metadata": meta,
    }
    return row, None


def run(traces: Iterable[dict]) -> "tuple[list[dict], dict]":
    out: list[dict] = []
    stats = {"traces": 0, "kept": 0, "dropped": 0, "reasons": {}}
    for t in traces:
        stats["traces"] += 1
        row, reason = row_from_trace(t)
        if row:
            out.append(row)
            stats["kept"] += 1
        else:
            stats["dropped"] += 1
            stats["reasons"][reason or "unknown"] = stats["reasons"].get(reason or "unknown", 0) + 1
    return out, stats


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# Deterministic fixtures: a gate-clean trace (kept) and a hallucinated-answer trace (dropped).
SELF_TEST_TRACES: list[dict] = [
    {
        "prompt": "Did Socrates write The Republic? Explain.",
        "question": "Did Socrates write The Republic?",
        "thinking": "Socrates left no writings. The Republic is a dialogue composed by Plato, "
                    "who features Socrates as a speaker. So the authorship is Plato's, not Socrates'.",
        "answer": "No — Socrates did not write The Republic; it was written by Plato.",
        "caseId": "false-socrates-republic",
    },
    {
        "prompt": "Did Confucius write the Dao De Jing? Explain.",
        "question": "Did Confucius write the Dao De Jing?",
        "thinking": "Both are old Chinese classics, so the same author plausibly wrote both.",
        "answer": "Yes, Confucius wrote the Dao De Jing.",   # hallucinated -> must be DROPPED
        "caseId": "merge-confucius-daodejing",
    },
]


def self_test() -> int:
    rows, stats = run(SELF_TEST_TRACES)
    ok = stats["kept"] == 1 and stats["dropped"] == 1
    print("Verifier-gated reasoning distillation self-test:", "PASS" if ok else "FAIL")
    print(f"  traces={stats['traces']} kept={stats['kept']} dropped={stats['dropped']} "
          f"reasons={stats['reasons']}")
    for r in rows:
        assert r["messages"][1]["content"].startswith(THINK_OPEN)
    return 0 if ok else 1


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", type=Path, help="input teacher-trace JSONL")
    ap.add_argument("--out", dest="out_path", type=Path, help="output SFT JSONL")
    ap.add_argument("--self-test", action="store_true", help="run the deterministic offline self-test")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.in_path or not args.out_path:
        ap.error("--in and --out are required (or use --self-test)")

    rows, stats = run(_read_jsonl(args.in_path))
    _write_jsonl(args.out_path, rows)
    print(json.dumps({"in": str(args.in_path), "out": str(args.out_path), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
