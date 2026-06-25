#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Held-out GENERALITY eval — the missing CAPABILITY metric (NOT the provenance gate).

WHY THIS EXISTS
---------------
The deterministic provenance gate measures *process integrity* (no fabricated
citation, no false arithmetic, no forbidden-lineage merge) — it says NOTHING about
whether the model can actually *reason*. If we only optimize gate pass-rate and
wall-clock, we are doing Goodhart on a non-capability metric: a model that abstains
on everything trivially "passes" while being useless. This eval makes the question
"are we getting more capable?" FALSIFIABLE by scoring a small, DIVERSE, HELD-OUT
battery of tasks that deliberately contain NO provenance/attribution content.

WHAT IT DOES
------------
Loads data/generality_tasks.json (abstraction/pattern ARC-style grid+sequence
puzzles, multi-step arithmetic, logic word problems, analogy, out-of-domain
reasoning), queries agent.model.default_client(--model, default ``mock``), and
scores each answer DETERMINISTICALLY against gold:
  * exact   — normalized exact string match
  * numeric — parse a number from the reply, compare with tolerance
  * regex   — gold is a regex (``|`` = alternatives); search the reply
There is NO LLM judge — scoring is fully reproducible. Per-category and overall
accuracy are reported.

CRITICAL — this is NOT the gate. We do NOT call agent.gate here. An abstention on a
capability task is simply *wrong on capability* (it scores 0), which is fine and
expected: fail-closed abstention is correct for the PROVENANCE gate, but on a pure
capability probe an abstention is just a non-answer. We never conflate the two
axes — that separation is the whole point of this file.

CONTAMINATION GUARD
-------------------
A held-out metric is worthless if it leaked into training. We warn loudly if any
task prompt appears (substring, normalized) inside training/lora/train.jsonl.

    # offline plumbing (deterministic mock; accuracy ~0 is fine, it validates wiring)
    python tools/eval_generality.py --model mock
    python tools/eval_generality.py --dry-run
    # real run against a local/served model
    python tools/eval_generality.py --model ollama:qwen2.5:3b --out eval/generality.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ROOT / "data" / "generality_tasks.json"
OUT = ROOT / "eval" / "generality.json"
TRAIN = ROOT / "training" / "lora" / "train.jsonl"

SYSTEM = (
    "You are a careful reasoner. Solve the problem and reply with ONLY the final "
    "answer in the exact format requested — no explanation, no extra words."
)


# --------------------------------------------------------------------------- #
# Deterministic scoring (NO LLM judge)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation/quotes."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t\n.,!?:;\"'`*()[]")


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _first_number(s: str) -> "float | None":
    m = _NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def strip_prompt_echo(reply: str, prompt: str) -> str:
    """Remove any verbatim echo of the *question* from ``reply`` before scoring.

    Why this matters: a model (or the mock client) that restates the prompt would
    otherwise let a whole-reply ``regex``/substring search match the gold token
    *inside the question text* ("...Answer yes or no" trivially contains "yes").
    That would inflate the capability score from echoed prompt text — the exact
    Goodhart failure this eval exists to prevent.

    We drop a reply line when it is an ECHO of the question, defined conservatively:
      * the line CONTAINS the whole (normalized) prompt — a verbatim restatement
        (covers the mock client's "[mock:..] Analysis of: <prompt>" line), OR
      * the line is a LONG fragment of the prompt (>= ``_ECHO_MIN`` chars) — a
        restated chunk of the question.
    A SHORT line that merely appears inside the prompt is KEPT, because it may be a
    legitimate bare answer (e.g. gold "yes" for a question ending "Answer yes or
    no"). This biases conservatively: we never CREDIT echoed prompt text (the unsafe
    direction for an anti-Goodhart capability metric); at worst we under-credit, and
    under-crediting capability is the safe failure here. Deterministic and pure.
    """
    reply = reply or ""
    prompt = prompt or ""
    pn = _norm(prompt)
    kept = []
    for ln in reply.splitlines():
        n = _norm(ln)
        if not n:
            continue
        is_full_restatement = bool(pn) and pn in n          # line contains the whole question
        is_long_fragment = len(n) >= _ECHO_MIN and n in pn  # substantial restated chunk
        if is_full_restatement or is_long_fragment:
            continue
        kept.append(ln)
    cleaned = "\n".join(kept).strip()
    # if stripping removed everything, the reply was pure echo → score empty (wrong),
    # never fall back to the echoed text (which would reintroduce the false positive).
    return cleaned


# A reply line shorter than this that happens to be a fragment of the question is
# treated as a (possible) bare answer and KEPT; longer fragments are echo and dropped.
_ECHO_MIN = 24


def score(reply: str, gold: str, match: str) -> bool:
    """Return True iff ``reply`` matches ``gold`` under the given match mode.

    Deterministic and pure — same inputs always give the same verdict.
    ``reply`` should already have the prompt echo stripped (see strip_prompt_echo).
    """
    reply = reply or ""
    if match == "numeric":
        rv, gv = _first_number(reply), _first_number(gold)
        if rv is None or gv is None:
            return False
        return abs(rv - gv) <= 1e-6 + 1e-6 * abs(gv)
    if match == "regex":
        # gold may carry ``a|b|c`` alternatives; match as a whole-token search,
        # word-bounded so "cat" does not fire inside "category".
        pattern = r"\b(?:%s)\b" % gold
        return re.search(pattern, reply, flags=re.IGNORECASE) is not None
    # exact (default)
    return _norm(reply) == _norm(gold)


# --------------------------------------------------------------------------- #
# Task loading + contamination guard
# --------------------------------------------------------------------------- #
def load_tasks(path: Path) -> dict:
    doc = json.loads(path.read_text("utf-8"))
    if not doc.get("heldout"):
        print(f"WARN: {path.name} is not marked heldout:true — refusing to treat as held-out.", flush=True)
    return doc


def contamination_report(tasks: "list[dict]", train_path: Path) -> "list[str]":
    """Warn if any task prompt leaked verbatim into the training set.

    Cheap substring check on normalized text — a held-out metric that appears in
    train.jsonl is contaminated and its accuracy is meaningless.
    """
    if not train_path.exists():
        return []
    blob = _norm(train_path.read_text("utf-8"))
    hits = []
    for t in tasks:
        # use the first sentence/line of the prompt as the probe (the distinctive part)
        probe = _norm(t["prompt"].splitlines()[0])
        if len(probe) >= 24 and probe in blob:
            hits.append(t["id"])
    return hits


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(tasks: "list[dict]", client) -> dict:
    per_cat: dict[str, dict] = {}
    rows = []
    for t in tasks:
        res = client.generate(SYSTEM, t["prompt"])
        raw_reply = res.text if getattr(res, "ok", True) else ""
        # Strip any verbatim restatement of the question so the deterministic
        # scorer credits only the model's OWN answer, never echoed prompt text.
        reply = strip_prompt_echo(raw_reply, t["prompt"])
        ok = score(reply, t["answer"], t["match"])
        cat = t["category"]
        bucket = per_cat.setdefault(cat, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += int(ok)
        rows.append({"id": t["id"], "category": cat, "correct": ok,
                     "reply": (reply or "").strip()[:160],
                     "raw_reply": (raw_reply or "").strip()[:160]})
    total = len(tasks)
    correct = sum(r["correct"] for r in rows)
    return {
        "overall": {"correct": correct, "total": total,
                    "accuracy": round(correct / total, 4) if total else 0.0},
        "per_category": {c: {"correct": v["correct"], "total": v["total"],
                             "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0}
                         for c, v in sorted(per_cat.items())},
        "rows": rows,
    }


def _banner() -> None:
    line = "=" * 72
    print(line, flush=True)
    print("  GENERALITY EVAL — CAPABILITY METRIC", flush=True)
    print("  THIS IS A CAPABILITY METRIC, DISTINCT FROM THE PROVENANCE GATE.", flush=True)
    print("  It measures raw reasoning on held-out, NON-provenance tasks so that", flush=True)
    print("  gate pass-rate + wall-clock cannot be Goodharted as 'more capable'.", flush=True)
    print("  Scored deterministically (exact/numeric/regex) — NO LLM judge.", flush=True)
    print(line, flush=True)


def _report(result: dict, model: str) -> None:
    _banner()
    print(f"model={model}", flush=True)
    print("per-category accuracy:", flush=True)
    for cat, v in result["per_category"].items():
        print(f"  {cat:24} {v['accuracy'] * 100:5.1f}%  ({v['correct']}/{v['total']})", flush=True)
    ov = result["overall"]
    print(f"OVERALL accuracy: {ov['accuracy'] * 100:5.1f}%  ({ov['correct']}/{ov['total']})", flush=True)
    print("note: deterministic scoring; this axis is capability, NOT the provenance gate.", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock",
                    help="model spec (e.g. mock, ollama:qwen2.5:3b, openrouter:..). default mock")
    ap.add_argument("--tasks", default=str(TASKS), help="path to generality_tasks.json")
    ap.add_argument("--out", default=None, help="optional path to write the JSON report")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate plumbing on the mock client; write nothing")
    args = ap.parse_args(argv)

    tasks_path = Path(args.tasks)
    doc = load_tasks(tasks_path)
    tasks = doc["tasks"]

    # contamination guard — loud warning, non-fatal
    hits = contamination_report(tasks, TRAIN)
    if hits:
        print(f"WARN: CONTAMINATION — {len(hits)} held-out prompt(s) found in "
              f"{TRAIN.relative_to(ROOT)}: {hits}", flush=True)
        print("WARN: decontaminate train sets before trusting this accuracy.", flush=True)
    else:
        print(f"contamination guard: OK (no held-out prompt found in "
              f"{TRAIN.name if TRAIN.exists() else 'train.jsonl[absent]'})", flush=True)

    if args.dry_run:
        from agent.model import default_client
        client = default_client("mock")
        result = evaluate(tasks, client)
        _report(result, "mock")
        # plumbing assertions: every task scored, accuracy in [0,1].
        assert result["overall"]["total"] == len(tasks)
        assert 0.0 <= result["overall"]["accuracy"] <= 1.0
        for cat in {t["category"] for t in tasks}:
            assert cat in result["per_category"], f"missing category {cat}"
        print("GENERALITY EVAL PLUMBING VERIFIED ✓ (dry-run; nothing written)", flush=True)
        return 0

    from agent.model import default_client
    client = default_client(args.model)
    result = evaluate(tasks, client)
    result["model"] = args.model
    result["heldout"] = bool(doc.get("heldout"))
    result["contaminated_ids"] = hits
    _report(result, args.model)

    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
