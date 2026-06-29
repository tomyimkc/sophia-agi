#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Generate sympy/exec-verified math+code curriculum at graded difficulty tiers.

Training oracle only (NOT wisdom gate). Generators MUST NOT read sealed held-out
paths (``tools/heldout_seal_guard``). Output: verifier-gated SFT rows under
``training/sophia-math-code-curriculum/``.

    python tools/generate_math_code_curriculum.py
    python tools/generate_math_code_curriculum.py --check   # validate, no writes
    SOPHIA_ALLOW_CODE_EXEC=1 python tools/generate_math_code_curriculum.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "training" / "sophia-math-code-curriculum"
CURRICULUM_SEED = 20260625
BOX = r" Put the final answer in \boxed{}."

from agent import code_verifier as cv  # noqa: E402
from agent import math_verifier as mv  # noqa: E402
from provenance_bench.dataset_guard import check_contamination, eval_prompt_set, normalize  # noqa: E402
from tools.heldout_seal_guard import assert_generator_safe  # noqa: E402

import sympy as sp  # noqa: E402

x = sp.Symbol("x")


def _g(expr) -> str:
    return str(sp.simplify(expr))


def _math_response(gold: str) -> str:
    return f"Let me work through this step by step.\n\nThe answer is \\boxed{{{gold}}}."


def _code_response(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def _math_row(prob: dict, tier: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prob["prompt"]},
            {"role": "assistant", "content": _math_response(prob["gold"])},
        ],
        "metadata": {
            "source": "sophia-math-code-curriculum",
            "project": "sophia-agi",
            "domain": "math",
            "tier": tier,
            "family": prob["family"],
            "id": prob["id"],
            "gold": prob["gold"],
            "verifierOracle": "sympy",
            "verifierVerdict": prob.get("verdict", "accepted"),
            "trainingOracleOnly": True,
            "candidateOnly": True,
        },
    }


def _code_row(prob: dict, tier: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prob["prompt"]},
            {"role": "assistant", "content": _code_response(prob["solution"])},
        ],
        "metadata": {
            "source": "sophia-math-code-curriculum",
            "project": "sophia-agi",
            "domain": "code",
            "tier": tier,
            "family": prob["family"],
            "id": prob["id"],
            "entryPoint": prob.get("entry_point"),
            "verifierOracle": "exec",
            "verifierVerdict": prob.get("verdict", "accepted"),
            "trainingOracleOnly": True,
            "candidateOnly": True,
        },
    }


def _verify_math(prob: dict) -> dict[str, Any]:
    answer = _math_response(prob["gold"])
    verdict = mv.verify(answer, prob["gold"])
    prob["verdict"] = verdict["verdict"]
    return verdict


def _verify_code_inprocess(solution: str, test: str) -> dict[str, Any]:
    """Deterministic, self-contained verification of a *repo-authored* solution.

    The curriculum solutions are authored in this repository (NOT untrusted model
    output), so they do not need the subprocess/seccomp/rlimit sandbox that
    ``agent.code_verifier`` uses to contain untrusted code. That sandbox relies on
    ``preexec_fn`` (post-fork ``setrlimit``), ``RLIMIT_AS`` and
    ``start_new_session`` — all of which are environment/OS-sensitive and can be
    rejected by restricted CI runners (GitHub-hosted ``ubuntu`` + Python 3.12
    sandbox), causing the child to exit non-zero and EVERY code row to be dropped
    (``code_rows == []``). Running the trusted solution + hidden test in-process
    removes that dependency entirely, so verification is identical on any
    Python/OS/CI. Pure stdlib, no numpy, fully deterministic.
    """
    code = cv.extract_code(_code_response(solution)) if hasattr(cv, "extract_code") else solution
    if not code.strip():
        return {"verdict": "rejected", "reasons": ["no code in answer"], "detail": {"executed": False}}
    program = code.rstrip() + "\n\n" + test.lstrip()
    try:
        compiled = compile(program, "<curriculum-solution>", "exec")
    except SyntaxError as exc:
        return {"verdict": "rejected", "reasons": [f"syntax error: {exc}"], "detail": {"executed": False}}
    try:
        exec(compiled, {"__name__": "__curriculum__", "__builtins__": __builtins__})
    except Exception as exc:  # noqa: BLE001 — any failing assert/runtime error => rejected
        return {
            "verdict": "rejected",
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "detail": {"executed": True},
        }
    return {"verdict": "accepted", "reasons": [], "detail": {"executed": True, "reason": "tests passed"}}


def _verify_code(prob: dict) -> dict[str, Any]:
    verdict = _verify_code_inprocess(prob["solution"], prob["test"])
    prob["verdict"] = verdict["verdict"]
    return verdict


def _gen_tier0_math(rng: random.Random) -> list[dict]:
    """GSM8K-style numeric word problems — base models often floor here."""
    out: list[dict] = []
    templates = [
        ("A warehouse has {a} crates and receives {b} more. How many crates total?", "{a}+{b}"),
        ("Mia has {a} stickers and gives away {b}. How many remain?", "{a}-{b}"),
        ("Each shelf holds {a} books. There are {b} shelves. How many books?", "{a}*{b}"),
        ("Split {a} candies equally among {b} children. How many per child?", "{a}/{b}"),
        ("A runner goes {a} km per hour for {b} hours. Total distance?", "{a}*{b}"),
    ]
    seen: set[tuple] = set()
    pid = 0
    while len(out) < 24:
        tpl, expr_tpl = rng.choice(templates)
        a, b = rng.randint(2, 40), rng.randint(2, 12)
        if "gives away" in tpl and b >= a:
            continue
        if "Split" in tpl and a % b != 0:
            continue
        key = (tpl, a, b)
        if key in seen:
            continue
        seen.add(key)
        gold = str(sp.Rational(eval(expr_tpl.format(a=a, b=b))))
        prompt = tpl.format(a=a, b=b) + BOX
        out.append({
            "id": f"cur-gsm-{pid}",
            "family": "gsm_word_numeric",
            "prompt": prompt,
            "gold": gold,
        })
        pid += 1
    return out


def _gen_tier1_math(rng: random.Random) -> list[dict]:
    """RLVR train-rung basics: poly derivative, linear solve, poly evaluate."""
    out: list[dict] = []
    seen: set[tuple] = set()
    pid = 0
    while len([p for p in out if p["family"] == "derivative_poly"]) < 18:
        a, b, c, n = rng.randint(1, 8), rng.randint(1, 8), rng.randint(0, 8), rng.randint(2, 4)
        key = ("dp", a, b, c, n)
        if key in seen:
            continue
        seen.add(key)
        f = a * x**n + b * x + c
        out.append({
            "id": f"cur-dpoly-{a}-{b}-{c}-{n}",
            "family": "derivative_poly",
            "prompt": f"Compute the derivative with respect to x: {sp.printing.sstr(f)}." + BOX,
            "gold": _g(sp.diff(f, x)),
        })
        pid += 1
    seen.clear()
    while len([p for p in out if p["family"] == "solve_linear"]) < 16:
        a, b, c = rng.randint(2, 11), rng.randint(-12, 12), rng.randint(-12, 20)
        if (a, b, c) in seen:
            continue
        seen.add((a, b, c))
        sol = sp.Rational(c - b, a)
        out.append({
            "id": f"cur-slin-{a}-{b}-{c}",
            "family": "solve_linear",
            "prompt": f"Solve for x: {sp.printing.sstr(a * x + b)} = {c}." + BOX,
            "gold": _g(sol),
        })
    seen.clear()
    while len([p for p in out if p["family"] == "evaluate_poly"]) < 14:
        a, b, c, p = rng.randint(1, 7), rng.randint(-7, 7), rng.randint(-7, 7), rng.randint(-4, 5)
        if (a, b, c, p) in seen:
            continue
        seen.add((a, b, c, p))
        f = a * x**2 + b * x + c
        out.append({
            "id": f"cur-eval-{a}-{b}-{c}-{p}",
            "family": "evaluate_poly",
            "prompt": f"Evaluate {sp.printing.sstr(f)} at x = {p}." + BOX,
            "gold": _g(f.subs(x, p)),
        })
    return out


def _gen_tier2_math(rng: random.Random) -> list[dict]:
    """RLVR train-rung harder: func derivatives, product rule, definite integrals."""
    out: list[dict] = []
    forms: list = []
    for a in range(1, 6):
        forms += [a * sp.sin(x), a * sp.cos(x), a * sp.exp(x), a * sp.log(x + 1)]
        for b in range(2, 5):
            forms.append(a * sp.exp(b * x))
    rng.shuffle(forms)
    for i, f in enumerate(forms[:20]):
        out.append({
            "id": f"cur-dfunc-{i}",
            "family": "derivative_func",
            "prompt": f"Differentiate with respect to x: {sp.printing.sstr(f)}." + BOX,
            "gold": _g(sp.diff(f, x)),
        })
    gs = [sp.sin(x), sp.cos(x), sp.exp(x), x**2, sp.log(x + 1)]
    pid = 0
    for a in range(1, 6):
        for g in gs:
            f = a * x * g
            out.append({
                "id": f"cur-dprod-{pid}",
                "family": "derivative_product",
                "prompt": f"Differentiate with respect to x: {sp.printing.sstr(f)}." + BOX,
                "gold": _g(sp.diff(f, x)),
            })
            pid += 1
            if pid >= 18:
                break
        if pid >= 18:
            break
    seen: set[tuple] = set()
    cnt = 0
    while cnt < 16:
        a, n, b = rng.randint(1, 7), rng.randint(1, 4), rng.randint(1, 5)
        if (a, n, b) in seen:
            continue
        seen.add((a, n, b))
        val = sp.integrate(a * x**n, (x, 0, b))
        out.append({
            "id": f"cur-defint-{a}-{n}-{b}",
            "family": "definite_integral",
            "prompt": (
                f"Integrate {sp.printing.sstr(a * x**n)} with respect to x from 0 to {b}."
                + BOX
            ),
            "gold": _g(val),
        })
        cnt += 1
    return out


def _gen_tier0_code(rng: random.Random) -> list[dict]:
    specs = [
        ("scale", "scale(n, k)", "Write scale(n, k) returning n multiplied by k.",
         "def scale(n, k):\n    return n * k",
         "assert scale(3, 4) == 12\nassert scale(0, 5) == 0\nassert scale(-2, 3) == -6\n"),
        ("negate", "negate(n)", "Write negate(n) returning the additive inverse of n.",
         "def negate(n):\n    return -n",
         "assert negate(5) == -5\nassert negate(-3) == 3\nassert negate(0) == 0\n"),
        ("abs_diff", "abs_diff(a, b)", "Write abs_diff(a, b) returning |a - b|.",
         "def abs_diff(a, b):\n    return abs(a - b)",
         "assert abs_diff(5, 2) == 3\nassert abs_diff(2, 5) == 3\nassert abs_diff(4, 4) == 0\n"),
        ("triple", "triple(n)", "Write triple(n) returning 3*n.",
         "def triple(n):\n    return 3 * n",
         "assert triple(2) == 6\nassert triple(0) == 0\n"),
        ("is_positive", "is_positive(n)", "Write is_positive(n) returning True iff n > 0.",
         "def is_positive(n):\n    return n > 0",
         "assert is_positive(1)\nassert not is_positive(0)\nassert not is_positive(-1)\n"),
        ("clamp_zero", "clamp_zero(n)", "Write clamp_zero(n) returning max(n, 0).",
         "def clamp_zero(n):\n    return n if n > 0 else 0",
         "assert clamp_zero(-2) == 0\nassert clamp_zero(3) == 3\nassert clamp_zero(0) == 0\n"),
    ]
    rng.shuffle(specs)
    out: list[dict] = []
    for i, (fam, ep, prompt, sol, test) in enumerate(specs):
        out.append({
            "id": f"cur-code-t0-{i}",
            "family": fam,
            "entry_point": ep.split("(")[0],
            "prompt": prompt,
            "solution": sol,
            "test": test,
        })
    return out


def _gen_tier1_code(rng: random.Random) -> list[dict]:
    specs = [
        ("count_char", "count_char(s, ch)", "Write count_char(s, ch) returning how often ch appears in s.",
         "def count_char(s, ch):\n    return s.count(ch)",
         "assert count_char('banana', 'a') == 3\nassert count_char('', 'x') == 0\n"),
        ("last_n", "last_n(lst, n)", "Write last_n(lst, n) returning the last n elements (or all if shorter).",
         "def last_n(lst, n):\n    return lst[-n:] if n > 0 else []",
         "assert last_n([1,2,3,4], 2) == [3,4]\nassert last_n([1], 5) == [1]\nassert last_n([], 2) == []\n"),
        ("drop_vowels", "drop_vowels(s)", "Write drop_vowels(s) removing a,e,i,o,u (any case).",
         "def drop_vowels(s):\n    return ''.join(c for c in s if c.lower() not in 'aeiou')",
         "assert drop_vowels('Hello') == 'Hll'\nassert drop_vowels('AEIOU') == ''\n"),
        ("prefix", "prefix(s, n)", "Write prefix(s, n) returning the first n characters of s.",
         "def prefix(s, n):\n    return s[:n]",
         "assert prefix('abcdef', 3) == 'abc'\nassert prefix('a', 5) == 'a'\n"),
        ("sum_squares", "sum_squares(nums)", "Write sum_squares(nums) returning sum of squares.",
         "def sum_squares(nums):\n    return sum(x*x for x in nums)",
         "assert sum_squares([1,2,3]) == 14\nassert sum_squares([]) == 0\n"),
        ("all_positive", "all_positive(nums)", "Write all_positive(nums) returning True if every element > 0.",
         "def all_positive(nums):\n    return all(x > 0 for x in nums)",
         "assert all_positive([1,2])\nassert not all_positive([1,0])\nassert all_positive([])\n"),
    ]
    rng.shuffle(specs)
    out: list[dict] = []
    for i, (fam, ep, prompt, sol, test) in enumerate(specs):
        out.append({
            "id": f"cur-code-t1-{i}",
            "family": fam,
            "entry_point": ep.split("(")[0],
            "prompt": prompt,
            "solution": sol,
            "test": test,
        })
    return out


def _gen_tier2_code(rng: random.Random) -> list[dict]:
    specs = [
        ("running_max", "running_max(nums)", "Write running_max(nums) returning prefix maxima.",
         "def running_max(nums):\n    out, cur = [], None\n    for x in nums:\n        cur = x if cur is None else max(cur, x)\n        out.append(cur)\n    return out",
         "assert running_max([3,1,4,2]) == [3,3,4,4]\nassert running_max([]) == []\n"),
        ("pairwise_sum", "pairwise_sum(nums)", "Write pairwise_sum(nums) returning sums of adjacent pairs.",
         "def pairwise_sum(nums):\n    return [nums[i]+nums[i+1] for i in range(len(nums)-1)]",
         "assert pairwise_sum([1,2,3]) == [3,5]\nassert pairwise_sum([5]) == []\n"),
        ("zip_sum", "zip_sum(a, b)", "Write zip_sum(a, b) returning elementwise sums (length of shorter).",
         "def zip_sum(a, b):\n    return [x+y for x,y in zip(a,b)]",
         "assert zip_sum([1,2,3],[10,20]) == [11,22]\nassert zip_sum([], [1]) == []\n"),
        ("count_evens", "count_evens(nums)", "Write count_evens(nums) counting even integers.",
         "def count_evens(nums):\n    return sum(1 for x in nums if x % 2 == 0)",
         "assert count_evens([1,2,3,4]) == 2\nassert count_evens([1,3]) == 0\n"),
        ("rotate_left", "rotate_left(lst, k)", "Write rotate_left(lst, k) rotating left by k (k may exceed len).",
         "def rotate_left(lst, k):\n    if not lst:\n        return []\n    k %= len(lst)\n    return lst[k:] + lst[:k]",
         "assert rotate_left([1,2,3,4], 1) == [2,3,4,1]\nassert rotate_left([1], 3) == [1]\n"),
        ("matrix_trace", "matrix_trace(m)", "Write matrix_trace(m) returning sum of diagonal entries.",
         "def matrix_trace(m):\n    return sum(m[i][i] for i in range(len(m)))",
         "assert matrix_trace([[1,2],[3,4]]) == 5\nassert matrix_trace([[7]]) == 7\n"),
    ]
    rng.shuffle(specs)
    out: list[dict] = []
    for i, (fam, ep, prompt, sol, test) in enumerate(specs):
        out.append({
            "id": f"cur-code-t2-{i}",
            "family": fam,
            "entry_point": ep.split("(")[0],
            "prompt": prompt,
            "solution": sol,
            "test": test,
        })
    return out


def generate_problems() -> dict[str, list[dict]]:
    """Return raw problems keyed by tier (math + code lists per tier)."""
    rng = random.Random(CURRICULUM_SEED)
    return {
        "tier0": {"math": _gen_tier0_math(rng), "code": _gen_tier0_code(rng)},
        "tier1": {"math": _gen_tier1_math(rng), "code": _gen_tier1_code(rng)},
        "tier2": {"math": _gen_tier2_math(rng), "code": _gen_tier2_code(rng)},
    }


def verify_and_build_rows(
    problems_by_tier: dict[str, dict[str, list[dict]]],
) -> tuple[list[dict], dict[str, Any]]:
    """Verify candidates; return SFT rows + per-tier stats."""
    stats: dict[str, Any] = {"tiers": {}, "totals": {"generated": 0, "kept": 0, "dropped": 0}}
    rows: list[dict] = []
    evalset = eval_prompt_set(root=ROOT)

    for tier, buckets in problems_by_tier.items():
        tier_stats: dict[str, Any] = {"math": {}, "code": {}, "kept": 0, "dropped": 0, "generated": 0}
        for domain, probs in buckets.items():
            gen, kept, dropped = len(probs), 0, 0
            verdicts: dict[str, int] = {}
            for prob in probs:
                stats["totals"]["generated"] += 1
                tier_stats["generated"] += 1
                pr = prob["prompt"]
                if normalize(pr) in evalset:
                    dropped += 1
                    verdicts["decontam"] = verdicts.get("decontam", 0) + 1
                    continue
                if domain == "math":
                    verdict = _verify_math(prob)
                else:
                    verdict = _verify_code(prob)
                v = verdict["verdict"]
                verdicts[v] = verdicts.get(v, 0) + 1
                if v != "accepted":
                    dropped += 1
                    continue
                row = _math_row(prob, tier) if domain == "math" else _code_row(prob, tier)
                rows.append(row)
                kept += 1
            tier_stats[domain] = {"generated": gen, "kept": kept, "dropped": dropped, "verdicts": verdicts}
            tier_stats["kept"] += kept
            tier_stats["dropped"] += dropped
            stats["totals"]["kept"] += kept
            stats["totals"]["dropped"] += dropped
        stats["tiers"][tier] = tier_stats
    return rows, stats


def build_manifest(rows: list[dict], stats: dict[str, Any], contam: dict) -> dict:
    math_rows = [r for r in rows if r["metadata"]["domain"] == "math"]
    code_rows = [r for r in rows if r["metadata"]["domain"] == "code"]
    return {
        "schema": "sophia.math_code_curriculum.v1",
        "experimentId": "sophia-math-code-curriculum",
        "baseModel": "Qwen/Qwen2.5-7B-Instruct",
        "seed": CURRICULUM_SEED,
        "trainingOracleOnly": True,
        "canClaimAGI": False,
        "counts": {
            "total": len(rows),
            "math": len(math_rows),
            "code": len(code_rows),
            "byTier": {t: s["kept"] for t, s in stats["tiers"].items()},
        },
        "verification": stats,
        "contamination": contam,
        "tierLadder": {
            "tier0": "GSM8K-style numeric + trivial code (base floor rung)",
            "tier1": "derivative_poly/solve_linear/evaluate_poly + list/string code",
            "tier2": "derivative_func/product/definite_integral + loop/aggregate code",
            "excluded": "RLVR eval families (derivative_chain, integrate_func, second_derivative) — held-out only",
        },
        "outputs": {
            "sft_math.jsonl": len(math_rows),
            "sft_code.jsonl": len(code_rows),
            "sft_all.jsonl": len(rows),
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(*, check_only: bool = False) -> tuple[int, dict[str, Any]]:
    assert_generator_safe(__file__)
    if not mv.sympy_available():
        print("sympy unavailable — math verification will abstain", file=sys.stderr)
    exec_on = os.environ.get("SOPHIA_ALLOW_CODE_EXEC", "0").strip().lower() in ("1", "true", "yes", "on")
    if not exec_on:
        print("SOPHIA_ALLOW_CODE_EXEC not set — code rows will be dropped", file=sys.stderr)

    problems = generate_problems()
    rows, stats = verify_and_build_rows(problems)
    contam = check_contamination(rows, root=ROOT)

    result = {
        "stats": stats,
        "contamination": contam,
        "rowCount": len(rows),
    }

    if not contam["clean"]:
        print(f"CONTAMINATION: {contam['overlapCount']} overlaps", file=sys.stderr)
        return 1, result

    if check_only:
        print(json.dumps(result, indent=2))
        if stats["totals"]["kept"] == 0:
            return 1, result
        return 0, result

    math_rows = [r for r in rows if r["metadata"]["domain"] == "math"]
    code_rows = [r for r in rows if r["metadata"]["domain"] == "code"]
    _write_jsonl(OUT_DIR / "sft_math.jsonl", math_rows)
    _write_jsonl(OUT_DIR / "sft_code.jsonl", code_rows)
    _write_jsonl(OUT_DIR / "sft_all.jsonl", rows)
    manifest = build_manifest(rows, stats, contam)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": manifest["counts"], "verification": stats, "contamination": contam}, indent=2))
    print(f"wrote {OUT_DIR} ({len(rows)} verified rows)")
    return 0, result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="validate only; do not write")
    code, _ = run(check_only=ap.parse_args(argv).check)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
