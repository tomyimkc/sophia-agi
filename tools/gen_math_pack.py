#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate the math RLVR problem pack with sympy-verified golds + a fixed split.

Design choice that matters for an HONEST reward: ``math_equivalent`` checks
symbolic *equivalence*, so "factor/expand/simplify" families are gameable — the
unsimplified input is equivalent to the answer, so a model that just restates the
question scores +1. We therefore build the pack ONLY from families where the
answer is a genuinely different object than the input (derivative, integral,
solution value, evaluation): restating the question is NOT equivalent to the
answer, so it scores 0. The reward is then a real signal.

Split is by FAMILY and is FIXED (not seed-dependent): a designated set of held-out
families (chain rule, function antiderivatives, second derivative) never appears in
training, so every seed shares the same held-out eval set and the seeds are
comparable. Golds are computed AND self-checked with sympy + the actual
``math_equivalent`` verifier, so a wrong gold can never ship.

    python tools/gen_math_pack.py            # regenerate provenance_bench/data/math_problems.json
    python tools/gen_math_pack.py --check    # generate in-memory and validate only
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

OUT = ROOT / "provenance_bench" / "data" / "math_problems.json"

import sympy as sp  # noqa: E402

x = sp.Symbol("x")
BOX = r" Put the final answer in \boxed{}."


def _g(expr) -> str:
    """Canonical sympy string for a gold answer."""
    return str(sp.simplify(expr))


def gen():
    """Return (train_list, eval_list) of problem dicts. Deterministic (seeded)."""
    rng = random.Random(20260624)
    train: list[dict] = []
    eval_: list[dict] = []

    def add(bucket, fam, pid, prompt, gold_expr):
        bucket.append({"id": pid, "family": fam, "split": "train" if bucket is train else "eval",
                       "prompt": prompt + BOX, "gold": _g(gold_expr)})

    # ---- TRAIN families (answer != question) ----
    # 1. derivative of a polynomial
    seen = set()
    while len([p for p in train if p["family"] == "derivative_poly"]) < 32:
        a, b, c, n = rng.randint(1, 6), rng.randint(1, 6), rng.randint(0, 6), rng.randint(2, 4)
        f = a * x**n + b * x + c
        key = ("dp", a, b, c, n)
        if key in seen:
            continue
        seen.add(key)
        add(train, "derivative_poly", f"dpoly-{a}-{b}-{c}-{n}",
            f"Differentiate with respect to x: {sp.printing.sstr(f)}.", sp.diff(f, x))

    # 2. derivative of a basic function (a*sin, a*cos, a*exp(bx), a*ln)
    forms = []
    for a in range(1, 5):
        forms += [a * sp.sin(x), a * sp.cos(x), a * sp.exp(x), a * sp.log(x)]
        for b in range(2, 4):
            forms.append(a * sp.exp(b * x))
    rng.shuffle(forms)
    for i, f in enumerate(forms[:28]):
        add(train, "derivative_func", f"dfunc-{i}",
            f"Differentiate with respect to x: {sp.printing.sstr(f)}.", sp.diff(f, x))

    # 3. derivative via product rule (x * g(x))
    gs = [sp.sin(x), sp.cos(x), sp.exp(x), x**2, sp.log(x)]
    pid = 0
    for a in range(1, 6):
        for g in gs:
            f = a * x * g
            add(train, "derivative_product", f"dprod-{pid}",
                f"Differentiate with respect to x: {sp.printing.sstr(f)}.", sp.diff(f, x))
            pid += 1
            if pid >= 25:
                break
        if pid >= 25:
            break

    # 4. solve a linear equation a*x + b = c  (gold: the value of x)
    cnt = 0
    seen = set()
    while cnt < 30:
        a, b, c = rng.randint(2, 9), rng.randint(-9, 9), rng.randint(-9, 18)
        if (a, b, c) in seen:
            continue
        seen.add(key := (a, b, c))
        sol = sp.Rational(c - b, a)
        add(train, "solve_linear", f"slin-{a}-{b}-{c}",
            f"Solve for x: {sp.printing.sstr(a*x + b)} = {c}.", sol)
        cnt += 1

    # 5. evaluate a polynomial at a point (gold: a number)
    cnt = 0
    seen = set()
    while cnt < 28:
        a, b, c, p = rng.randint(1, 6), rng.randint(-6, 6), rng.randint(-6, 6), rng.randint(-3, 4)
        if (a, b, c, p) in seen:
            continue
        seen.add((a, b, c, p))
        f = a * x**2 + b * x + c
        add(train, "evaluate_poly", f"eval-{a}-{b}-{c}-{p}",
            f"Evaluate {sp.printing.sstr(f)} at x = {p}.", f.subs(x, p))
        cnt += 1

    # 6. definite integral of a monomial over [0, b]  (gold: a number)
    cnt = 0
    seen = set()
    while cnt < 25:
        a, n, b = rng.randint(1, 6), rng.randint(1, 4), rng.randint(1, 4)
        if (a, n, b) in seen:
            continue
        seen.add((a, n, b))
        val = sp.integrate(a * x**n, (x, 0, b))
        add(train, "definite_integral", f"defint-{a}-{n}-{b}",
            f"Compute the definite integral of {sp.printing.sstr(a*x**n)} with respect to x from 0 to {b}.", val)
        cnt += 1

    # ---- EVAL (held-out) families: genuine generalizations, never trained ----
    # 7. chain rule: d/dx sin(a*x**2), cos(a*x**2), exp(a*x**2), (a*x+b)**n
    pid = 0
    bases = []
    for a in range(1, 4):
        bases += [sp.sin(a * x**2), sp.cos(a * x**2), sp.exp(a * x**2)]
        for b in range(1, 3):
            for n in range(2, 4):
                bases.append((a * x + b)**n)
    rng.shuffle(bases)
    for f in bases[:20]:
        add(eval_, "derivative_chain", f"dchain-{pid}",
            f"Differentiate with respect to x: {sp.printing.sstr(f)}.", sp.diff(f, x))
        pid += 1

    # 8. antiderivative of a basic function (omit the constant)
    pid = 0
    funcs = []
    for a in range(1, 5):
        funcs += [a * sp.cos(x), a * sp.sin(x), a * sp.exp(x)]
        for b in range(2, 4):
            funcs.append(a * sp.exp(b * x))
    rng.shuffle(funcs)
    for f in funcs[:20]:
        add(eval_, "integrate_func", f"ifunc-{pid}",
            f"Find the antiderivative with respect to x (omit the constant): {sp.printing.sstr(f)}.",
            sp.integrate(f, x))
        pid += 1

    # 9. second derivative of a polynomial
    pid = 0
    seen = set()
    while pid < 20:
        a, b, c, n = rng.randint(1, 6), rng.randint(1, 6), rng.randint(0, 6), rng.randint(2, 5)
        if (a, b, c, n) in seen:
            continue
        seen.add((a, b, c, n))
        f = a * x**n + b * x**2 + c
        add(eval_, "second_derivative", f"d2-{a}-{b}-{c}-{n}",
            f"Find the second derivative with respect to x: {sp.printing.sstr(f)}.", sp.diff(f, x, 2))
        pid += 1

    return train, eval_


def validate(problems: list[dict]) -> list[str]:
    """Every gold must self-score +1 through the REAL verifier."""
    from provenance_bench.math_reward import reward_for_problem
    bad = []
    for p in problems:
        score, _ = reward_for_problem(r"\boxed{" + p["gold"] + "}", p["gold"])
        if score != 1.0:
            bad.append(p["id"])
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; do not write")
    args = ap.parse_args(argv)

    train, eval_ = gen()
    problems = train + eval_
    bad = validate(problems)
    tf = sorted({p["family"] for p in train})
    ef = sorted({p["family"] for p in eval_})
    assert set(tf).isdisjoint(ef), "train/eval families must be disjoint"
    print(f"train: {len(train)} ({len(tf)} families) | eval: {len(eval_)} ({len(ef)} families)")
    print(f"train families: {tf}")
    print(f"eval  families: {ef}")
    if bad:
        print(f"GOLD SELF-CHECK FAILED for {len(bad)}: {bad[:10]}")
        return 1
    print("all golds self-score +1 ✓")
    if args.check:
        return 0

    out = {
        "_meta": {
            "description": "Symbolic-math RLVR pack (sympy-generated, self-verified golds). Families chosen "
                           "so the answer != the question (derivative/integral/solution/evaluation), making "
                           "math_equivalent a non-gameable reward. Split is by FIXED held-out families "
                           "(identical across seeds) so seeds are comparable; eval families are genuine "
                           "generalizations (chain rule, function antiderivatives, second derivative) never "
                           "seen in training. Regenerate with tools/gen_math_pack.py.",
            "schemaVersion": 2,
            "trainFamilies": tf,
            "evalFamilies": ef,
            "counts": {"train": len(train), "eval": len(eval_)},
        },
        "problems": problems,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(problems)} problems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
