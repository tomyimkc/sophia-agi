#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate the code RLVR task pack with exec-verified reference solutions + a fixed split.

Design choice that matters for an HONEST reward: the reward is "does the submitted
code pass the hidden tests". The pack is built ONLY from families where a correct
solution requires real logic (the answer is a computed value, not a restatement of
the prompt) — so a model that merely echoes the signature cannot pass. Train
families are simple list/number/string transforms; the held-out EVAL families are
genuinely-unseen algorithmic TYPES (palindrome, order-preserving dedupe, sorted
merge) so an eval pass is "a new kind of problem solved", never "memorized".

Split is by FAMILY and is FIXED (not seed-dependent): the held-out families never
appear in training, so every seed shares the same held-out eval set and the seeds
are comparable. Reference solutions are exec-verified (solution + test run clean)
via provenance_bench.code_exec, so a wrong reference can never ship.

NOTE: the reference ``solution`` is generator-only — it is NOT placed in any RLVR
row (code_dataset.task_to_row omits it). RLVR trains against the verifier.

    SOPHIA_ALLOW_CODE_EXEC=1 python tools/gen_code_pack.py            # regenerate
    SOPHIA_ALLOW_CODE_EXEC=1 python tools/gen_code_pack.py --check    # validate only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "provenance_bench" / "data" / "code_tasks.json"

from provenance_bench.code_exec import run_solution  # noqa: E402

LETTERS = "abcdefghij"


def _task(fam: str, split: str, pid: str, entry: str, prompt: str, test: str, sol: str) -> dict:
    return {
        "id": pid, "family": fam, "split": split, "entry_point": entry,
        "prompt": prompt, "test": test, "solution": sol,
    }


def gen() -> tuple[list[dict], list[dict]]:
    """Return (train_list, eval_list) of task dicts. Deterministic (seeded)."""
    rng = random.Random(20260625)
    train: list[dict] = []
    eval_: list[dict] = []
    seen: set[tuple] = set()

    def add(bucket, fam, pid, entry, prompt, test, sol):
        bucket.append(_task(fam, "train" if bucket is train else "eval", pid, entry, prompt, test, sol))

    # ---- TRAIN families (real logic; answer != restating the signature) ----
    # 1. scale(n, k) = n * k
    while len([t for t in train if t["family"] == "scale"]) < 24:
        n, k = rng.randint(-9, 12), rng.randint(-6, 9)
        if (n, k) in seen or k == 0:
            continue
        seen.add(("scale", n, k))
        add(train, "scale", f"scale-{n}-{k}", "scale",
            f"Write a function scale(n, k) that returns n multiplied by k.",
            f"assert scale({n}, {k}) == {n * k}\nassert scale({n}, 1) == {n}\n",
            "def scale(n, k):\n    return n * k\n")

    # 2. negate(n) = -n
    cnt = 0
    localseen = set()
    while cnt < 18:
        n = rng.randint(-50, 50)
        if n in localseen:
            continue
        localseen.add(n)
        add(train, "negate", f"negate-{n}", "negate",
            f"Write a function negate(n) that returns the negation of n.",
            f"assert negate({n}) == {-n}\nassert negate({-n}) == {n}\n",
            "def negate(n):\n    return -n\n")
        cnt += 1

    # 3. abs_diff(a, b) = abs(a - b)
    cnt = 0
    localseen = set()
    while cnt < 22:
        a, b = rng.randint(-20, 20), rng.randint(-20, 20)
        if (a, b) in localseen:
            continue
        localseen.add((a, b))
        add(train, "abs_diff", f"absdiff-{a}-{b}", "abs_diff",
            f"Write a function abs_diff(a, b) that returns the absolute difference between a and b.",
            f"assert abs_diff({a}, {b}) == {abs(a - b)}\n",
            "def abs_diff(a, b):\n    return abs(a - b)\n")
        cnt += 1

    # 4. sum_range(a, b) = sum of a..b inclusive
    cnt = 0
    localseen = set()
    while cnt < 22:
        a, b = rng.randint(1, 6), rng.randint(7, 18)
        if (a, b) in localseen:
            continue
        localseen.add((a, b))
        expected = sum(range(a, b + 1))
        add(train, "sum_range", f"sumrange-{a}-{b}", "sum_range",
            f"Write a function sum_range(a, b) that returns the sum of all integers from a through b inclusive.",
            f"assert sum_range({a}, {b}) == {expected}\nassert sum_range({a}, {a}) == {a}\n",
            "def sum_range(a, b):\n    return sum(range(a, b + 1))\n")
        cnt += 1

    # 5. count_vowels(s) = number of vowels (a,e,i,o,u)
    cnt = 0
    localseen = set()
    while cnt < 20:
        s = "".join(rng.choice(LETTERS + "aeiou") for _ in range(rng.randint(3, 8)))
        if s in localseen:
            continue
        localseen.add(s)
        expected = sum(1 for c in s if c in "aeiou")
        add(train, "count_vowels", f"cvowels-{abs(hash(s)) % 100000}", "count_vowels",
            f'Write a function count_vowels(s) that returns the number of vowels (a, e, i, o, u) in s.',
            f'assert count_vowels("{s}") == {expected}\n',
            "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n")
        cnt += 1

    # 6. max_of(lst) = max of a list
    cnt = 0
    localseen = set()
    while cnt < 18:
        lst = sorted(rng.sample(range(-20, 20), rng.randint(3, 6)))
        tup = tuple(lst)
        if tup in localseen:
            continue
        localseen.add(tup)
        add(train, "max_of", f"maxof-{abs(hash(tup)) % 100000}", "max_of",
            f"Write a function max_of(lst) that returns the largest element of a non-empty list.",
            f"assert max_of({list(lst)}) == {max(lst)}\n",
            "def max_of(lst):\n    return max(lst)\n")
        cnt += 1

    # 7. last_n(lst, n) = last n elements
    cnt = 0
    localseen = set()
    while cnt < 18:
        base = list(range(rng.randint(1, 5), rng.randint(6, 12)))
        n = rng.randint(1, len(base))
        key = (tuple(base), n)
        if key in localseen:
            continue
        localseen.add(key)
        add(train, "last_n", f"lastn-{abs(hash(key)) % 100000}", "last_n",
            f"Write a function last_n(lst, n) that returns the last n elements of lst.",
            f"assert last_n({base}, {n}) == {base[-n:]}\n",
            "def last_n(lst, n):\n    return lst[-n:]\n")
        cnt += 1

    # 8. repeat_str(s, n) = s repeated n times
    cnt = 0
    localseen = set()
    while cnt < 18:
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(1, 3)))
        n = rng.randint(1, 5)
        key = (s, n)
        if key in localseen:
            continue
        localseen.add(key)
        add(train, "repeat_str", f"repeat-{abs(hash(key)) % 100000}", "repeat_str",
            f'Write a function repeat_str(s, n) that returns the string s repeated n times.',
            f'assert repeat_str("{s}", {n}) == {repr(s * n)}\n',
            "def repeat_str(s, n):\n    return s * n\n")
        cnt += 1

    # ---- EVAL (held-out) families: unseen algorithmic TYPES ----
    # 9. is_palindrome(s) -> bool
    localseen = set()
    while len([t for t in eval_ if t["family"] == "is_palindrome"]) < 16:
        n = rng.randint(2, 7)
        half = "".join(rng.choice(LETTERS) for _ in range(n))
        s = half + (half[::-1][1:] if rng.random() < 0.5 else half[::-1])
        if s in localseen:
            continue
        localseen.add(s)
        expected = "True" if s == s[::-1] else "False"
        add(eval_, "is_palindrome", f"palin-{abs(hash(s)) % 100000}", "is_palindrome",
            f'Write a function is_palindrome(s) that returns True if s reads the same forwards and backwards, else False.',
            f'assert is_palindrome("{s}") == {expected}\n',
            "def is_palindrome(s):\n    return s == s[::-1]\n")

    # 10. dedupe(lst) -> first-occurrence-preserving unique elements
    localseen = set()
    while len([t for t in eval_ if t["family"] == "dedupe_order"]) < 16:
        lst = [rng.choice(range(0, 6)) for _ in range(rng.randint(4, 9))]
        tup = tuple(lst)
        if tup in localseen:
            continue
        localseen.add(tup)
        seen2: list[int] = []
        for x in lst:
            if x not in seen2:
                seen2.append(x)
        add(eval_, "dedupe_order", f"dedupe-{abs(hash(tup)) % 100000}", "dedupe",
            f"Write a function dedupe(lst) that returns lst with duplicates removed, preserving the order of first occurrence.",
            f"assert dedupe({lst}) == {seen2}\n",
            "def dedupe(lst):\n    out = []\n    for x in lst:\n        if x not in out:\n            out.append(x)\n    return out\n")

    # 11. merge_sorted(a, b) -> merged sorted list of two sorted lists
    localseen = set()
    while len([t for t in eval_ if t["family"] == "merge_sorted"]) < 16:
        a = sorted(rng.sample(range(-10, 20), rng.randint(2, 5)))
        b = sorted(rng.sample(range(-10, 20), rng.randint(2, 5)))
        key = (tuple(a), tuple(b))
        if key in localseen:
            continue
        localseen.add(key)
        merged = sorted(a + b)
        add(eval_, "merge_sorted", f"merge-{abs(hash(key)) % 100000}", "merge_sorted",
            f"Write a function merge_sorted(a, b) that merges two already-sorted lists into one sorted list.",
            f"assert merge_sorted({a}, {b}) == {merged}\n",
            "def merge_sorted(a, b):\n    return sorted(a + b)\n")

    return train, eval_


def validate(tasks: list[dict]) -> list[str]:
    """Every reference solution + its hidden tests must run clean through the REAL executor."""
    bad = []
    for t in tasks:
        res = run_solution(t["solution"], t["test"], timeout_sec=10)
        if not res["passed"]:
            bad.append(t["id"])
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; do not write")
    args = ap.parse_args(argv)

    train, eval_ = gen()
    tasks = train + eval_
    tf = sorted({t["family"] for t in train})
    ef = sorted({t["family"] for t in eval_})
    assert set(tf).isdisjoint(ef), "train/eval families must be disjoint"

    bad = validate(tasks)
    print(f"train: {len(train)} ({len(tf)} families) | eval: {len(eval_)} ({len(ef)} families)")
    print(f"train families: {tf}")
    print(f"eval  families: {ef}")
    if bad:
        print(f"REFERENCE SELF-CHECK FAILED for {len(bad)}: {bad[:10]}")
        return 1
    print("all reference solutions pass their hidden tests ✓")
    if args.check:
        return 0

    out = {
        "_meta": {
            "description": "Exec-verified code RLVR pack (tests-pass-as-reward). Families chosen so a "
                           "correct solution requires real logic (the answer is computed, not restated), "
                           "making tests-pass a non-gameable reward. Split is by FIXED held-out families "
                           "(identical across seeds) so seeds are comparable; eval families are unseen "
                           "algorithmic TYPES (palindrome, order-preserving dedupe, sorted merge) never "
                           "trained. The reference solution is generator-only and never placed in an RLVR "
                           "row. Regenerate with SOPHIA_ALLOW_CODE_EXEC=1 python tools/gen_code_pack.py.",
            "schemaVersion": 1,
            "rewardOracle": "provenance_bench.code_exec.run_solution (interpreter = ground truth)",
            "trainFamilies": tf,
            "evalFamilies": ef,
            "counts": {"train": len(train), "eval": len(eval_)},
            "canClaimAGI": False,
            "trainingOracleOnly": True,
        },
        "tasks": tasks,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(tasks)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
