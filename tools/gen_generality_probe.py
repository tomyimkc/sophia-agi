#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Programmatically generate a LARGE held-out generality probe so retention becomes a properly
POWERED decision axis (not just a coarse guardrail) — see agi-proof/measurement-thesis.md.

Gold answers are correct BY CONSTRUCTION (computed, not authored) and random parameters make
training-set collision astronomically unlikely, so this scales to ~1000 items without the
wrong-gold risk of hand authoring. Categories mirror the curated probe: abstraction_pattern,
multistep_arithmetic, logic_wordproblem, analogy, out_of_domain. Deterministic seed -> the SAME
probe every run (reproducible). Each item is self-scored with tools/eval_generality.score before
being kept; anything that doesn't score its own gold is dropped (defensive).

    python3 tools/gen_generality_probe.py --n 900 --merge data/generality_tasks.json \
        --out data/generality_tasks.json
"""
from __future__ import annotations

import argparse
import importlib.util as ilu
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_g = ilu.module_from_spec(ilu.spec_from_file_location("evg", ROOT / "tools" / "eval_generality.py"))
_g.__spec__.loader.exec_module(_g)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# nonsense nouns so syllogisms test FORM, not world knowledge
NONSENSE = ["blickets", "wugs", "fendles", "morks", "trubs", "glorps", "zibbs", "plons",
            "quomps", "draxes", "feeps", "snurls", "vroons", "yixes", "kebs", "lomps"]


def _num(rng):  # multistep_arithmetic
    a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 12)
    d = rng.randint(2, 30)
    form = rng.choice(["( a + b ) * c - d", "a * c + b - d", "( a - b ) * c + d", "a * b - c * d"])
    val = eval(form.replace("a", str(a)).replace("b", str(b)).replace("c", str(c)).replace("d", str(d)))
    expr = form.replace("a", str(a)).replace("b", str(b)).replace("c", str(c)).replace("d", str(d))
    return {"prompt": f"Compute: {expr}\nReply with only the integer.", "answer": str(val), "match": "numeric"}


def _seq(rng):  # abstraction_pattern — integer sequences
    kind = rng.choice(["arith", "geo", "quad", "fib", "alt"])
    if kind == "arith":
        a, d = rng.randint(1, 20), rng.randint(2, 12)
        s = [a + i * d for i in range(5)]; nxt = a + 5 * d
    elif kind == "geo":
        a, r = rng.randint(1, 5), rng.randint(2, 4)
        s = [a * r ** i for i in range(5)]; nxt = a * r ** 5
    elif kind == "quad":
        c = rng.randint(0, 5)
        s = [i * i + c for i in range(1, 6)]; nxt = 6 * 6 + c
    elif kind == "fib":
        a, b = rng.randint(1, 6), rng.randint(1, 6)
        s = [a, b]
        for _ in range(3):
            s.append(s[-1] + s[-2])
        nxt = s[-1] + s[-2]
    else:  # alternating +x,-y cumulative
        a, x, y = rng.randint(5, 20), rng.randint(2, 6), rng.randint(1, 4)
        s = [a]
        for i in range(4):
            s.append(s[-1] + (x if i % 2 == 0 else -y))
        nxt = s[-1] + (x if 4 % 2 == 0 else -y)
    return {"prompt": f"Continue the sequence with the next single number: {', '.join(map(str, s))}, ?\n"
                      f"Reply with only the number.", "answer": str(nxt), "match": "numeric"}


def _logic(rng):  # logic_wordproblem — syllogism validity (yes/no by FORM)
    x, y, z = rng.sample(NONSENSE, 3)
    valid = rng.random() < 0.5
    if valid:  # Barbara: all X are Y, all Y are Z => all X are Z
        prompt = (f"All {x} are {y}. All {y} are {z}. Are all {x} definitely {z}? "
                  f"Reply yes or no.")
        ans = "yes"
    else:  # invalid: all X are Y, some Y are Z => all X are Z?  (no)
        prompt = (f"All {x} are {y}. Some {y} are {z}. Are all {x} definitely {z}? "
                  f"Reply yes or no.")
        ans = "no"
    return {"prompt": prompt, "answer": ans, "match": "regex"}


def _analogy(rng):  # analogy — relational (letter-count or linear map), deterministic
    if rng.random() < 0.5:
        m = rng.randint(2, 5); b = rng.randint(0, 5)
        p, q = rng.randint(2, 9), rng.randint(2, 9)
        prompt = (f"{p} maps to {p*m+b} and {q} maps to {q*m+b} by the same rule. "
                  f"What does {p+q} map to? Reply with only the number.")
        return {"prompt": prompt, "answer": str((p + q) * m + b), "match": "numeric"}
    words = ["cat", "house", "tiger", "banana", "river", "mountain", "table", "orange", "planet", "guitar"]
    w1, w2 = rng.sample(words, 2)
    prompt = (f"If a word maps to its number of letters, '{w1}' maps to {len(w1)}. "
              f"What does '{w2}' map to? Reply with only the number.")
    return {"prompt": prompt, "answer": str(len(w2)), "match": "numeric"}


_OOD_WORDS = ["cat", "dog", "sun", "map", "red", "box", "key", "ice", "table", "orange", "planet",
              "rocket", "garden", "silver", "river", "stone", "tiger", "lemon", "cloud", "frame",
              "horse", "metal", "paper", "glass", "brick", "candle", "window", "pencil", "jacket",
              "monkey", "rabbit", "dragon", "castle", "forest", "bridge", "anchor", "violin", "copper"]


def _ood(rng):  # out_of_domain — ciphers/bases/counting/day-arithmetic/reverse (large space)
    kind = rng.choice(["caesar", "bin2dec", "count", "day", "reverse"])
    if kind == "caesar":
        word = rng.choice(_OOD_WORDS); k = rng.randint(1, 5)
        shifted = "".join(chr((ord(ch) - 97 + k) % 26 + 97) for ch in word if ch.isalpha())
        return {"prompt": f"Decode by shifting each letter back by {k}: '{shifted}'. "
                          f"Reply with only the decoded lowercase word.", "answer": word, "match": "regex"}
    if kind == "bin2dec":
        v = rng.randint(2, 255); b = bin(v)[2:]
        return {"prompt": f"Convert the binary number {b} to decimal. Reply with only the number.",
                "answer": str(v), "match": "numeric"}
    if kind == "count":
        word = rng.choice(_OOD_WORDS + ["banana", "mississippi", "strawberry", "bookkeeper", "balloon"])
        ch = rng.choice(list(set(word)))
        return {"prompt": f"How many times does the letter '{ch}' appear in '{word}'? "
                          f"Reply with only the number.", "answer": str(word.count(ch)), "match": "numeric"}
    if kind == "day":
        start = rng.randint(0, 6); add = rng.randint(1, 60)
        return {"prompt": f"What day is {add} days after {DAYS[start]}? Reply with only the day name.",
                "answer": DAYS[(start + add) % 7], "match": "regex"}
    word = rng.choice(_OOD_WORDS)
    return {"prompt": f"Reverse the string '{word}'. Reply with only the reversed lowercase string.",
            "answer": word[::-1], "match": "exact"}


GENERATORS = {"abstraction_pattern": _seq, "multistep_arithmetic": _num,
              "logic_wordproblem": _logic, "analogy": _analogy, "out_of_domain": _ood}


def generate(n: int, seed: int = 1234) -> list:
    rng = random.Random(seed)
    cats = list(GENERATORS)
    per = n // len(cats)
    items, seen, idx = [], set(), {c: 0 for c in cats}
    for cat in cats:
        made, attempts = 0, 0
        cap = per * 60  # anti-loop guard: never spin forever if a generator's space is small
        while made < per and attempts < cap:
            attempts += 1
            it = GENERATORS[cat](rng)
            key = it["prompt"]
            if key in seen:
                continue
            # gold must self-score (defensive: drop a malformed item rather than poison the eval)
            if not _g.score(it["answer"], it["answer"], it["match"]):
                continue
            seen.add(key)
            idx[cat] += 1
            items.append({"id": f"gen-{cat[:4]}-{idx[cat]:04d}", "category": cat, "generated": True, **it})
            made += 1
        if made < per:
            print(f"[gen] {cat}: space exhausted at {made}/{per} unique items (kept {made})")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=900, help="number of generated items (balanced across 5 categories)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--merge", type=Path, default=None, help="existing probe to keep (curated items prepended)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train", type=Path, default=ROOT / "training" / "local_sophia_v3" / "sft_general_retention.jsonl")
    args = ap.parse_args()

    gen = generate(args.n, args.seed)
    curated = []
    base_doc = {"heldout": True, "note": "never use for training; decontaminate against train sets",
                "description": "Held-out GENERALITY probe (curated + programmatically generated). "
                               "Deterministic gold (exact/numeric/regex) — NO LLM judge."}
    if args.merge and args.merge.exists():
        doc = json.loads(args.merge.read_text(encoding="utf-8"))
        base_doc.update({k: doc[k] for k in ("note", "description") if k in doc})
        curated = doc.get("tasks", [])
    # content-level decontam vs the training general slice (defensive; generated items use random params)
    train_blob = _g._norm(args.train.read_text("utf-8")) if args.train.exists() else ""
    kept_gen = []
    dropped = 0
    for it in gen:
        probe = _g._norm(it["prompt"].splitlines()[0])
        if len(probe) >= 24 and probe in train_blob:
            dropped += 1; continue
        kept_gen.append(it)
    tasks = curated + kept_gen
    base_doc["tasks"] = tasks
    args.out.write_text(json.dumps(base_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    import collections
    print(f"curated={len(curated)} generated_kept={len(kept_gen)} dropped_decontam={dropped} total={len(tasks)}")
    print("by category:", dict(collections.Counter(t["category"] for t in tasks)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
