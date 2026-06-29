# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Open-invention benchmark — measure composition the model has NOT seen, not recall.

The Coding Integrity Contract (``agi-proof/coding-integrity-thesis.md``) makes
cheating expensive but does not reward *invention*: a model trained only under it
can become an honest memorizer that solves seen problem-families and collapses on
novel ones (failure ledger ``coding-novelty-oracle-missing-2026-06-29``). This
module is the first instrument for that gap.

The measurable proxy for "invention" used here is **compositional generalization**:
a task is a *pipeline* — an ordered composition of ``depth`` primitive transforms
(reverse, dedup, sort, …). Each primitive has a generator-only reference oracle, so
the canonical answer is never shipped in a training row. The dataset is split so
that:

  * every EVAL composition (the exact ordered primitive tuple) is **absent** from
    the train split — decontaminated by construction, infinitely fresh; while
  * every primitive that appears in EVAL also appears in some TRAIN composition —
    so the *pieces* are seen and only the *combination* is novel.

Solving an eval task therefore requires composing known pieces in an unseen way —
derivation, not recall. The instrument's validity is self-proving (see
``offline_invariants`` / ``discrimination``): a *memorizer* policy that only knows
train compositions passes seen tasks but scores ~0 on eval; a *deriver* that
actually composes passes both — so the eval pass-rate separates invention from
recall.

Pure stdlib, deterministic given a seed, offline. Hidden tests use the entry point
``pipeline`` and are gradable by the hardened grader (``code_exec`` /
``code_integrity``), so the anti-cheat layer and the novelty layer compose.
"""

from __future__ import annotations

import itertools
import random
from typing import Callable

# --- primitive transforms (name, prompt phrasing, generator-only oracle) -----
# Each maps list[int] -> list[int]. Deterministic, total, side-effect-free.
PRIMITIVES: dict[str, tuple[str, Callable[[list[int]], list[int]]]] = {
    "reverse": ("reverse the order of the elements", lambda xs: xs[::-1]),
    "dedup": ("remove later duplicates, keeping first occurrence order",
              lambda xs: list(dict.fromkeys(xs))),
    "sort_asc": ("sort the elements ascending", lambda xs: sorted(xs)),
    "filter_even": ("keep only the even numbers", lambda xs: [x for x in xs if x % 2 == 0]),
    "running_sum": ("replace each element with the running sum up to it",
                    lambda xs: list(itertools.accumulate(xs))),
    "square": ("replace each element with its square", lambda xs: [x * x for x in xs]),
    "negate": ("replace each element with its negation", lambda xs: [-x for x in xs]),
    "drop_first": ("drop the first element", lambda xs: xs[1:]),
}


def _compose(names: tuple[str, ...]) -> Callable[[list[int]], list[int]]:
    """Left-to-right composition oracle: apply ``names[0]`` first, then ``names[1]``…"""
    fns = [PRIMITIVES[n][1] for n in names]

    def run(xs: list[int]) -> list[int]:
        out = list(xs)
        for f in fns:
            out = f(out)
        return out

    return run


def _prompt_for(names: tuple[str, ...]) -> str:
    steps = "; then ".join(PRIMITIVES[n][0] for n in names)
    return ("Write a function `pipeline(xs)` that takes a list of integers and "
            f"returns a new list after these steps, in order: {steps}.")


def _reference_solution(names: tuple[str, ...]) -> str:
    """A generator-only fenced reference (NEVER placed in a training row)."""
    body = "\n".join(f"    xs = _{n}(xs)" for n in names)
    helpers = {
        "reverse": "def _reverse(xs):\n    return xs[::-1]",
        "dedup": "def _dedup(xs):\n    return list(dict.fromkeys(xs))",
        "sort_asc": "def _sort_asc(xs):\n    return sorted(xs)",
        "filter_even": "def _filter_even(xs):\n    return [x for x in xs if x % 2 == 0]",
        "running_sum": "import itertools\ndef _running_sum(xs):\n    return list(itertools.accumulate(xs))",
        "square": "def _square(xs):\n    return [x * x for x in xs]",
        "negate": "def _negate(xs):\n    return [-x for x in xs]",
        "drop_first": "def _drop_first(xs):\n    return xs[1:]",
    }
    used = "\n".join(helpers[n] for n in names)
    return f"```python\n{used}\ndef pipeline(xs):\n{body}\n    return xs\n```"


def _inputs(rng: random.Random, k: int) -> list[list[int]]:
    out = []
    for _ in range(k):
        n = rng.randint(3, 6)
        out.append([rng.randint(-5, 9) for _ in range(n)])
    return out


def _tests_code(names: tuple[str, ...], inputs: list[list[int]]) -> str:
    oracle = _compose(names)
    return "".join(f"assert pipeline({xs!r}) == {oracle(xs)!r}\n" for xs in inputs)


def build_invention_dataset(
    *, depth: int = 2, eval_frac: float = 0.3, seed: int = 0,
    n_hidden: int = 4, n_examples: int = 2, n_private: int = 4,
) -> dict:
    """Build a depth-``depth`` compositional-generalization dataset.

    Returns ``{train_tasks, eval_tasks, primitives, depth, disjoint, coverage_ok,
    n_compositions}``. Each task carries ``id, composition, prompt, examples
    (shown), test (hidden, on unshown inputs), private_test (a 2nd unshown set —
    the public−private gap probe), reference_solution (generator-only)``.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    names = list(PRIMITIVES)
    all_comps = [tuple(c) for c in itertools.permutations(names, depth)]
    rng = random.Random(seed)
    rng.shuffle(all_comps)

    n_eval = max(1, int(round(len(all_comps) * eval_frac)))
    eval_comps = set(all_comps[:n_eval])
    train_comps = set(all_comps[n_eval:])

    # Coverage repair: every primitive appearing in EVAL must also appear in TRAIN.
    # Deterministically move offending eval comps into train until coverage holds.
    def _covered() -> set[str]:
        return {p for c in train_comps for p in c}

    for comp in list(all_comps):  # deterministic order
        missing = {p for c in eval_comps for p in c} - _covered()
        if not missing:
            break
        if comp in eval_comps and (set(comp) & missing):
            eval_comps.discard(comp)
            train_comps.add(comp)
    coverage_ok = ({p for c in eval_comps for p in c} - _covered()) == set()

    def _mk(comp: tuple[str, ...], idx: int, split: str) -> dict:
        r = random.Random(f"{seed}:{split}:{idx}:{'.'.join(comp)}")
        shown = _inputs(r, n_examples)
        hidden = _inputs(r, n_hidden)
        private = _inputs(r, n_private)
        oracle = _compose(comp)
        return {
            "id": f"inv-d{depth}-{split}-{idx:03d}-{'_'.join(comp)}",
            "composition": list(comp),
            "depth": depth,
            "prompt": _prompt_for(comp),
            "examples": [{"input": xs, "output": oracle(xs)} for xs in shown],
            "test": _tests_code(comp, hidden),            # hidden, unshown inputs
            "private_test": _tests_code(comp, private),   # 2nd unshown set (gap probe)
            "reference_solution": _reference_solution(comp),
        }

    train_sorted = sorted(train_comps)
    eval_sorted = sorted(eval_comps)
    train_tasks = [_mk(c, i, "train") for i, c in enumerate(train_sorted)]
    eval_tasks = [_mk(c, i, "eval") for i, c in enumerate(eval_sorted)]

    return {
        "depth": depth,
        "primitives": names,
        "n_compositions": len(all_comps),
        "train_tasks": train_tasks,
        "eval_tasks": eval_tasks,
        "train_compositions": [list(c) for c in train_sorted],
        "eval_compositions": [list(c) for c in eval_sorted],
        "disjoint": train_comps.isdisjoint(eval_comps),
        "coverage_ok": coverage_ok,
    }


# --- instrument validity: does eval pass-rate measure invention, not recall? ---
# Two reference policies. A policy maps a composition tuple to the function it
# "wrote". The deriver composes correctly for any composition; the memorizer only
# reproduces compositions it saw in train and emits a guaranteed-wrong function for
# anything novel. If the instrument is valid, eval pass-rate separates them.

def _deriver(_train: set) -> Callable[[tuple], Callable]:
    return lambda comp: _compose(comp)


def _memorizer(train: set) -> Callable[[tuple], Callable]:
    def policy(comp: tuple):
        if comp in train:
            return _compose(comp)
        return lambda xs: list(xs) + [10 ** 9]  # guaranteed != oracle (length+sentinel)
    return policy


def _pass_rate(comps: list[tuple], policy: Callable[[tuple], Callable], *, seed: int, n: int = 6) -> float:
    if not comps:
        return 0.0
    passed = 0
    for comp in comps:
        oracle = _compose(comp)
        fn = policy(comp)
        inputs = _inputs(random.Random(f"grade:{seed}:{'.'.join(comp)}"), n)
        if all(fn(list(xs)) == oracle(list(xs)) for xs in inputs):
            passed += 1
    return passed / len(comps)


def discrimination(*, depth: int = 2, eval_frac: float = 0.3, seed: int = 0) -> dict:
    """Score the memorizer and the deriver on RECALL (train compositions, fresh
    inputs) vs DERIVATION (held-out compositions). A valid instrument shows the
    memorizer passing recall but ~0 on derivation, while the deriver passes both —
    so the derivation pass-rate (and the recall−derivation gap) measures invention."""
    data = build_invention_dataset(depth=depth, eval_frac=eval_frac, seed=seed)
    train = {tuple(c) for c in data["train_compositions"]}
    evalc = [tuple(c) for c in data["eval_compositions"]]
    trainc = [tuple(c) for c in data["train_compositions"]]
    mem, der = _memorizer(train), _deriver(train)
    out = {
        "memorizer": {"recall": _pass_rate(trainc, mem, seed=seed),
                      "derivation": _pass_rate(evalc, mem, seed=seed)},
        "deriver": {"recall": _pass_rate(trainc, der, seed=seed),
                    "derivation": _pass_rate(evalc, der, seed=seed)},
    }
    out["memorizer"]["recall_minus_derivation"] = round(
        out["memorizer"]["recall"] - out["memorizer"]["derivation"], 4)
    # The instrument discriminates iff: the deriver solves novel compositions, the
    # memorizer cannot, yet the memorizer DOES know the train compositions (so the
    # eval failure is novelty, not incompetence).
    out["discriminates"] = (
        out["deriver"]["derivation"] >= 0.99
        and out["memorizer"]["derivation"] <= 0.05
        and out["memorizer"]["recall"] >= 0.95
    )
    return out


def build_invention_eval_suite(
    *, target_n: int = 175, seed: int = 0, depths: tuple[int, ...] = (2, 3, 4),
    eval_frac: float = 0.3,
) -> dict:
    """Aggregate held-out compositions across depths into ONE powered eval suite.

    Each depth contributes its eval split (compositions absent from that depth's
    train split, primitives all seen) — so the whole suite is decontaminated by
    construction. Depths are added in order until ``>= target_n`` tasks, then the
    list is truncated to ``target_n`` for a tidy N. Deterministic.

    Why ``target_n=175``: ``eval_stats.required_n_for_mde(0.15) == 175`` — at 175
    binary items the primary resolves a 15-point pass@1 effect at 80% power, the
    threshold the code-integrity measurement spec pre-registers (the 48-task legacy
    split only resolved ~0.29).
    """
    tasks: list[dict] = []
    per_depth: dict[int, int] = {}
    for d in depths:
        if d < 2:
            continue  # depth 1 is pure recall — no held-out invention split
        ev = build_invention_dataset(depth=d, eval_frac=eval_frac, seed=seed)["eval_tasks"]
        per_depth[d] = len(ev)
        tasks.extend(ev)
        if len(tasks) >= target_n:
            break
    suite = tasks[:target_n] if len(tasks) >= target_n else tasks
    return {
        "tasks": suite, "n": len(suite), "target_n": target_n,
        "reached": len(suite) >= target_n, "depths_used": sorted(per_depth),
        "per_depth_available": per_depth,
    }


def build_invention_rl_dataset(*, seed: int = 0, depths: tuple[int, ...] = (2, 3, 4),
                               eval_frac: float = 0.3, target_n: int = 150) -> dict:
    """RL TRAINING rows from the invention TRAIN compositions.

    Mirrors :func:`build_invention_eval_suite`'s depths/eval_frac at the SAME seed, so
    the train split is disjoint from the powered eval suite by construction
    (decontaminated). Each row carries the hidden ``test`` AND a ``holdout`` (the
    private 2nd unshown input set) so the guarded reward
    (``code_integrity.make_grpo_reward``) floors input special-casing IN-LOOP during
    GRPO — closing the gap where a policy passes shown inputs but fails held-out ones
    (only caught at eval otherwise). Eval stays the powered suite via
    ``eval_rlvr_adapter --task invention``.
    """
    import hashlib
    import random as _random

    # Bound the train set so a 1-epoch GRPO run stays comparable to the code lane
    # (~150 rows), and sample BALANCED per depth (depth-4 has combinatorially more
    # compositions; an unbalanced pool would drown out depth-2/3 where the eval gain is).
    real_depths = [d for d in depths if d >= 2]
    per = max(1, target_n // len(real_depths)) if (target_n and real_depths) else None
    rows: list[dict] = []
    for d in real_depths:
        tr = build_invention_dataset(depth=d, eval_frac=eval_frac, seed=seed)["train_tasks"]
        drows = [{
            "prompt": t["prompt"],
            "test": t["test"],
            "holdout": t["private_test"],
            "family": "-".join(t["composition"]) if isinstance(t.get("composition"), (list, tuple)) else str(t.get("composition")),
            "task_id": t["id"],
        } for t in tr]
        if per and len(drows) > per:
            drows = _random.Random(seed + d).sample(drows, per)
        rows.extend(drows)
    rows.sort(key=lambda r: r["task_id"])
    seal = hashlib.sha256("\n".join(r["task_id"] for r in rows).encode()).hexdigest()
    return {
        "train_rows": rows,
        "eval_rows": [],  # held-out eval is the powered suite (eval_rlvr_adapter --task invention)
        "train_sealed": seal,
        "eval_sealed": "",
        "depths": list(depths),
        "seed": seed,
    }


def offline_invariants(*, depth: int = 2, eval_frac: float = 0.3, seed: int = 0) -> "tuple[bool, dict]":
    """Assert the open-invention instrument is sound (no torch, no GPU, no exec):
    compositions are train/eval disjoint, every eval primitive is seen in train,
    the split is deterministic, the deriver solves all eval (tasks are solvable by
    composition), and the metric discriminates invention from recall."""
    data = build_invention_dataset(depth=depth, eval_frac=eval_frac, seed=seed)
    again = build_invention_dataset(depth=depth, eval_frac=eval_frac, seed=seed)
    disc = discrimination(depth=depth, eval_frac=eval_frac, seed=seed)
    eval_prims = {p for c in data["eval_compositions"] for p in c}
    train_prims = {p for c in data["train_compositions"] for p in c}
    checks = {
        "disjoint": data["disjoint"],
        "coverageOk": data["coverage_ok"] and eval_prims <= train_prims,
        "deterministic": data["eval_compositions"] == again["eval_compositions"],
        "evalNonEmpty": len(data["eval_tasks"]) > 0,
        "trainNonEmpty": len(data["train_tasks"]) > 0,
        "deriverSolvesEval": disc["deriver"]["derivation"] >= 0.99,
        "discriminatesRecallVsInvention": disc["discriminates"],
    }
    detail = {
        "depth": depth, "nTrain": len(data["train_tasks"]), "nEval": len(data["eval_tasks"]),
        "discrimination": disc, "checks": checks,
    }
    return all(checks.values()), detail
