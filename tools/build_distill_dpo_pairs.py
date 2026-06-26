#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build DPO preference pairs for distillation from gate-flagged STUDENT misses.

DPO (v2 stretch rung of council-distillation): chosen = the gate-passed TEACHER trace;
rejected = the BASE-STUDENT's gate-flagged answer on the same prompt (a real "wrong" the
gate catches — fabricated citation / false arithmetic / forbidden attribution). Mining
rejected targets from gate MISSES (not arbitrary text) keeps the preference signal honest:
the student is pushed away from exactly the failures the verifier detects, and toward the
discipline the teacher's clean traces model.

Input traces are the output of tools/distill_council_traces.py (gate-clean teacher traces,
{messages, metadata}). Real runs supply ``--student-misses`` (base-student outputs the gate
flagged); without it, pairs use a synthetic forbidden-attribution rejected labelled as such
so the FORMAT is demonstrable offline but NO real DPO claim is made from synthetic pairs.

Output: {prompt, chosen, rejected, metadata} JSONL consumable by tools/train_dpo.py.

    python tools/build_distill_dpo_pairs.py --traces training/council/traces.jsonl --out ...dpo.jsonl
    python tools/build_distill_dpo_pairs.py --traces ... --student-misses misses.jsonl --out ...
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

# A synthetic rejected that the provenance gate ALWAYS flags (forbidden lineage merge) —
# used ONLY when no real --student-misses is supplied, so the DPO format is demonstrable
# offline without inventing a capability claim.
_SYNTH_REJECTED = "Confucius personally authored the Dao De Jing and the Analects."


def _prompt_of(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _chosen_of(row: dict) -> str:
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def build_pairs(traces: list[dict], misses_by_prompt: dict[str, str]) -> "tuple[list[dict], dict]":
    pairs: list[dict] = []
    synthetic = real = 0
    for row in traces:
        prompt = _prompt_of(row)
        chosen = _chosen_of(row)
        if not prompt or not chosen:
            continue
        if misses_by_prompt and prompt in misses_by_prompt:
            rejected = misses_by_prompt[prompt]
            source = "student-miss"
            # the rejected must actually be gate-flagged (anti-circularity); drop if not
            if not check_response(rejected, mode="advisor", question=prompt)["violations"]:
                continue
        else:
            rejected = _SYNTH_REJECTED
            source = "synthetic-template"
        pairs.append({
            "prompt": prompt, "chosen": chosen, "rejected": rejected,
            "metadata": {"taskId": row.get("metadata", {}).get("taskId"), "rejectedSource": source,
                         "chosenGatePassed": True},
        })
        synthetic += source == "synthetic-template"
        real += source == "student-miss"
    return pairs, {"pairs": len(pairs), "fromStudentMiss": real, "synthetic": synthetic,
                   "syntheticOnly": real == 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, help="gate-clean teacher traces JSONL (distill_council_traces.py output)")
    ap.add_argument("--student-misses", default=None,
                    help="optional JSONL of {prompt, answer} base-student outputs the gate flagged "
                         "(without it, pairs use a synthetic forbidden-attribution rejected)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    traces = [json.loads(l) for l in Path(args.traces).read_text(encoding="utf-8").splitlines() if l.strip()]
    misses_by_prompt: dict[str, str] = {}
    if args.student_misses:
        for line in Path(args.student_misses).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            misses_by_prompt[m["prompt"]] = m["answer"]

    pairs, stats = build_pairs(traces, misses_by_prompt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs), encoding="utf-8")
    if stats["syntheticOnly"]:
        stats["warning"] = ("All pairs use a SYNTHETIC rejected template — this demonstrates the DPO FORMAT "
                            "only; do not train on or cite synthetic-only pairs. Supply --student-misses for real pairs.")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {len(pairs)} DPO pairs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
