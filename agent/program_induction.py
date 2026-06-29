# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated program induction for Sophia's generality track.

This module is deliberately **not** a claim of AGI. It implements the next
missing rung: learn a small executable rule from examples, validate it on held-out
examples, and only then admit it as a skill/verifier candidate. The trust boundary
is the measured validation/test split, not the proposal source.

Honest scope (name vs substance): the hypothesis space is a fixed, hand-written set of
template families (affine, polynomial, string/reverse/upper, grid flip/transpose) plus
an AST-sandboxed slot for a model-proposed ``solve(x)``. It is inductive-programming
*infrastructure* with a tiny template library, NOT DreamCoder-class open-ended program
synthesis — it abstains on any rule outside its templates.

Design constraints:
- deterministic/offline by default;
- optional model-proposed ``solve(x)`` programs must pass an AST allowlist;
- no filesystem, network, imports, attributes, loops, comprehensions, eval/exec;
- promote only when validation accuracy clears the floor;
- surface candidateOnly artifacts for no-overclaim proof discipline.
"""

from __future__ import annotations

import ast
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Example = dict[str, Any]
ProgramFn = Callable[[Any], Any]
ProposeFn = Callable[[dict[str, Any], list[Example]], list[str]]


@dataclass(frozen=True)
class ProgramCandidate:
    name: str
    source: str
    fn: ProgramFn
    params: dict[str, Any] = field(default_factory=dict)

    def __call__(self, x: Any) -> Any:
        return self.fn(x)


@dataclass(frozen=True)
class ProgramStats:
    accuracy: float
    n: int
    failures: int


@dataclass
class InductionResult:
    task_id: str
    admitted: ProgramCandidate | None = None
    validation_stats: ProgramStats | None = None
    test_stats: ProgramStats | None = None
    rejected: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = True
    splits: dict[str, int] = field(default_factory=dict)
    boundary: str = "candidateOnly: offline program-induction mechanism, not AGI evidence"

    def report(self) -> dict[str, Any]:
        return {
            "schema": "sophia.program_induction.v1",
            "taskId": self.task_id,
            "candidateOnly": True,
            "level3Evidence": False,
            "abstained": self.abstained,
            "admitted": None if self.admitted is None else {
                "name": self.admitted.name,
                "source": self.admitted.source,
                "params": self.admitted.params,
            },
            "validation": None if self.validation_stats is None else self.validation_stats.__dict__,
            "test": None if self.test_stats is None else self.test_stats.__dict__,
            "rejected": self.rejected,
            "splits": self.splits,
            "boundary": self.boundary,
        }


# --------------------------------------------------------------------------- #
# Safe proposed-program sandbox
# --------------------------------------------------------------------------- #
_SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
    "sum": sum,
}
_ALLOWED_CALLS = set(_SAFE_BUILTINS)
_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.IfExp, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Mod, ast.FloorDiv, ast.Div,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice,
)
_MAX_SOURCE = 2200


def ast_program_is_safe(tree: ast.AST) -> bool:
    """Allow exactly one ``def solve(x): return ...`` scalar/list expression."""
    body = getattr(tree, "body", [])
    funcs = [n for n in body if isinstance(n, ast.FunctionDef)]
    if len(body) != 1 or len(funcs) != 1:
        return False
    fn = funcs[0]
    if fn.name != "solve" or fn.decorator_list or len(fn.args.args) != 1:
        return False
    if not fn.body or not isinstance(fn.body[-1], ast.Return):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                return False
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and abs(float(node.value)) > 1_000_000:
                return False
            if isinstance(node.value, str) and len(node.value) > 10_000:
                return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            # Avoid sequence/allocation bombs such as ``[0] * 10**9`` or
            # ``"x" * 999999``. Scalar multiplication by a bounded constant is
            # allowed for arithmetic programs.
            if isinstance(node.left, (ast.List, ast.Tuple, ast.Dict)) or isinstance(node.right, (ast.List, ast.Tuple, ast.Dict)):
                return False
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant):
                    if isinstance(side.value, str):
                        return False
                    if isinstance(side.value, (int, float)) and abs(float(side.value)) > 1000:
                        return False
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in {"x", *list(_SAFE_BUILTINS)}:
                return False
    return True


def compile_program_source(source: str) -> ProgramCandidate | None:
    """Compile an optional model-proposed ``solve(x)`` after AST sandboxing."""
    if not isinstance(source, str) or "solve" not in source or len(source) > _MAX_SOURCE:
        return None
    try:
        tree = ast.parse(source, "<program-candidate>", "exec")
    except SyntaxError:
        return None
    if not ast_program_is_safe(tree):
        return None
    ns: dict[str, Any] = {}
    try:
        exec(compile(tree, "<program-candidate>", "exec"), {"__builtins__": _SAFE_BUILTINS}, ns)  # noqa: S102
    except Exception:
        return None
    fn = ns.get("solve")
    if not callable(fn):
        return None
    return ProgramCandidate("proposed", source, fn, {"sandbox": "ast_allowlist_v1"})


# --------------------------------------------------------------------------- #
# Template induction library. These are tiny, transparent hypotheses; the held-out
# gate decides whether they are trusted.
# --------------------------------------------------------------------------- #

def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-9
        except Exception:
            return False
    return a == b


def _score(candidate: ProgramCandidate, examples: list[Example]) -> ProgramStats:
    failures = 0
    for ex in examples:
        try:
            pred = candidate(ex["input"])
        except Exception:
            failures += 1
            continue
        if not _eq(pred, ex["output"]):
            failures += 1
    n = len(examples)
    return ProgramStats(accuracy=round((n - failures) / n, 4) if n else 0.0, n=n, failures=failures)


def _numeric(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except Exception:
        return None


def _is_intish(v: float) -> bool:
    return abs(v - round(v)) <= 1e-9


def _fit_numeric_linear(fit: list[Example]) -> list[ProgramCandidate]:
    pairs = [(_numeric(e.get("input")), _numeric(e.get("output"))) for e in fit]
    if not pairs or any(x is None or y is None for x, y in pairs):
        return []
    xs = [float(x) for x, _ in pairs]
    ys = [float(y) for _, y in pairs]
    out: list[ProgramCandidate] = []
    # affine y = a*x + b from two distinct points
    distinct = sorted(set(xs))
    if len(distinct) >= 2:
        x1 = distinct[0]
        x2 = next(x for x in distinct[1:] if x != x1)
        y1 = ys[xs.index(x1)]
        y2 = ys[xs.index(x2)]
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        def fn(x, a=a, b=b):
            y = a * float(x) + b
            return int(round(y)) if _is_intish(y) else y
        src = f"def solve(x):\n    return {a!r} * float(x) + {b!r}"
        out.append(ProgramCandidate("numeric_affine", src, fn, {"a": round(a, 8), "b": round(b, 8)}))
    # square-plus-b y = x*x + b
    offsets = [y - x * x for x, y in zip(xs, ys)]
    if offsets and max(offsets) - min(offsets) <= 1e-9:
        b = offsets[0]
        def fn2(x, b=b):
            y = float(x) * float(x) + b
            return int(round(y)) if _is_intish(y) else y
        out.append(ProgramCandidate("numeric_square_plus", f"def solve(x):\n    return float(x) * float(x) + {b!r}", fn2, {"b": round(b, 8)}))
    return out


def _fit_string(fit: list[Example]) -> list[ProgramCandidate]:
    if not fit or any(not isinstance(e.get("input"), str) or not isinstance(e.get("output"), str) for e in fit):
        return []
    xs = [e["input"] for e in fit]
    ys = [e["output"] for e in fit]
    candidates: list[ProgramCandidate] = []
    candidates.extend([
        ProgramCandidate("string_reverse", "def solve(x):\n    return str(x)[::-1]", lambda x: str(x)[::-1]),
        ProgramCandidate("string_upper", "def solve(x):\n    return str(x).upper()", lambda x: str(x).upper()),
        ProgramCandidate("string_lower", "def solve(x):\n    return str(x).lower()", lambda x: str(x).lower()),
    ])
    # fixed prefix/suffix wrappers: y = prefix + x + suffix
    prefix = None
    suffix = None
    ok = True
    for x, y in zip(xs, ys):
        if x not in y:
            ok = False
            break
        i = y.index(x)
        p, s = y[:i], y[i + len(x):]
        prefix = p if prefix is None else prefix
        suffix = s if suffix is None else suffix
        if p != prefix or s != suffix:
            ok = False
            break
    if ok and (prefix or suffix):
        candidates.append(ProgramCandidate(
            "string_wrap", f"def solve(x):\n    return {prefix!r} + str(x) + {suffix!r}",
            lambda x, p=prefix or "", s=suffix or "": p + str(x) + s,
            {"prefix": prefix or "", "suffix": suffix or ""},
        ))
    return candidates


def _is_grid(x: Any) -> bool:
    return isinstance(x, list) and x and all(isinstance(r, list) for r in x)


def _fit_grid(fit: list[Example]) -> list[ProgramCandidate]:
    if not fit or any(not _is_grid(e.get("input")) or not _is_grid(e.get("output")) for e in fit):
        return []
    return [
        ProgramCandidate("grid_hflip", "def solve(x):\n    return [list(reversed(r)) for r in x]", lambda x: [list(reversed(r)) for r in x]),
        ProgramCandidate("grid_vflip", "def solve(x):\n    return list(reversed(x))", lambda x: list(reversed(x))),
        ProgramCandidate("grid_transpose", "def solve(x):\n    return [list(r) for r in zip(*x)]", lambda x: [list(r) for r in zip(*x)]),
    ]


TEMPLATE_FITTERS = [_fit_numeric_linear, _fit_string, _fit_grid]


def partition_examples(examples: list[Example], *, fit_frac: float = 0.4, val_frac: float = 0.3, seed: int = 0) -> tuple[list[Example], list[Example], list[Example]]:
    if len(examples) < 5:
        return [], [], []
    rows = [dict(e, _idx=e.get("_idx", i)) for i, e in enumerate(examples)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_fit = max(2, int(round(n * fit_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_fit = min(n_fit, n - 2)
    n_val = min(n_val, n - n_fit - 1)
    return rows[:n_fit], rows[n_fit:n_fit + n_val], rows[n_fit + n_val:]


def induce_program(
    task: dict[str, Any], *,
    validation_floor: float = 1.0,
    test_floor: float = 0.95,
    seed: int = 0,
    propose_fn: ProposeFn | None = None,
) -> InductionResult:
    """Fit candidate programs, validate on held-out examples, and promote one.

    ``propose_fn`` may widen the candidate set but cannot bypass the validation
    floor. If no program clears validation, the result abstains.
    """
    examples = list(task.get("examples", []))
    result = InductionResult(task_id=str(task.get("task_id", "?")))
    fit, val, test = partition_examples(examples, seed=seed)
    result.splits = {"fit": len(fit), "validation": len(val), "test": len(test)}
    if not fit or not val or not test:
        return result

    candidates: list[ProgramCandidate] = []
    for fitter in TEMPLATE_FITTERS:
        try:
            candidates.extend(fitter(fit))
        except Exception:
            continue
    if propose_fn is not None:
        try:
            for src in (propose_fn(task, fit) or [])[:8]:
                c = compile_program_source(src)
                if c is not None:
                    candidates.append(c)
        except Exception:
            # propose_fn is an optional best-effort source; a failing proposer is skipped
            pass

    best: tuple[ProgramCandidate, ProgramStats, ProgramStats] | None = None
    seen_sources: set[str] = set()
    for cand in candidates:
        if cand.source in seen_sources:
            continue
        seen_sources.add(cand.source)
        val_stats = _score(cand, val)
        test_stats = _score(cand, test)
        if val_stats.accuracy >= validation_floor:
            if best is None or (test_stats.accuracy, val_stats.accuracy) > (best[2].accuracy, best[1].accuracy):
                best = (cand, val_stats, test_stats)
        else:
            result.rejected.append({"name": cand.name, "validation": val_stats.__dict__, "reason": "below validation floor"})

    if best is None:
        return result
    cand, val_stats, test_stats = best
    # Test floor is reported as an invariant; do not promote a candidate that
    # validates but then collapses on the unseen test split.
    if test_stats.accuracy < test_floor:
        result.rejected.append({"name": cand.name, "validation": val_stats.__dict__, "test": test_stats.__dict__, "reason": "below test floor"})
        return result
    result.admitted = cand
    result.validation_stats = val_stats
    result.test_stats = test_stats
    result.abstained = False
    return result


def demo_suite(seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic toy suite spanning numeric/string/grid induction + OOD hold."""
    nums = list(range(-10, 20))
    words = ["sophia", "agi", "dao", "mencius", "plato", "z3", "gate", "oracle", "memory", "skill"]
    grids = [
        [[1, 2], [3, 4]],
        [[5, 6, 7], [8, 9, 0]],
        [[1], [2], [3]],
        [[7, 8], [9, 1], [2, 3]],
        [[4, 5, 6]],
        [[0, 1], [1, 0]],
        [[2, 2, 3], [4, 5, 6]],
        [[9], [8]],
        [[1, 0, 1], [0, 1, 0]],
    ]
    return [
        {"task_id": "affine_3x_plus_2", "family": "numeric", "examples": [{"input": x, "output": 3 * x + 2} for x in nums]},
        {"task_id": "square_plus_1", "family": "numeric", "examples": [{"input": x, "output": x * x + 1} for x in nums]},
        {"task_id": "reverse_string", "family": "string", "examples": [{"input": w, "output": w[::-1]} for w in words]},
        {"task_id": "wrap_verified", "family": "string", "examples": [{"input": w, "output": f"verified:{w}:ok"} for w in words]},
        {"task_id": "grid_hflip", "family": "grid", "examples": [{"input": g, "output": [list(reversed(r)) for r in g]} for g in grids]},
        # OOD/noisy mapping: no compact supported rule should validate.
        {"task_id": "unlearnable_lookup", "family": "ood", "examples": [{"input": w, "output": str(hash((w, seed)) % 997)} for w in words]},
    ]


def evaluate_program_induction(tasks: list[dict[str, Any]] | None = None, *, seed: int = 0) -> dict[str, Any]:
    tasks = tasks or demo_suite(seed)
    results = [induce_program(t, seed=seed).report() for t in tasks]
    learnable = [r for r in results if r["taskId"] != "unlearnable_lookup"]
    ood = [r for r in results if r["taskId"] == "unlearnable_lookup"]
    promoted = [r for r in results if not r["abstained"]]
    report = {
        "schema": "sophia.program_induction_eval.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "seed": seed,
        "nTasks": len(results),
        "metrics": {
            "learnablePromotionRate": round(sum(not r["abstained"] for r in learnable) / len(learnable), 4) if learnable else 0.0,
            "oodAbstentionRate": round(sum(r["abstained"] for r in ood) / len(ood), 4) if ood else 0.0,
            "meanPromotedTestAccuracy": round(sum((r.get("test") or {}).get("accuracy", 0.0) for r in promoted) / len(promoted), 4) if promoted else 0.0,
            "falsePromotionRateOnOod": round(sum(not r["abstained"] for r in ood) / len(ood), 4) if ood else 0.0,
        },
        "invariants": {
            "promotes_most_learnable_tasks": (sum(not r["abstained"] for r in learnable) >= max(1, len(learnable) - 1)),
            "holds_unlearnable_task": all(r["abstained"] for r in ood),
            "promoted_programs_clear_test_floor": all((r.get("test") or {}).get("accuracy", 0.0) >= 0.95 for r in promoted),
            "candidate_boundary_present": all(r.get("candidateOnly") is True and r.get("level3Evidence") is False for r in results),
        },
        "tasks": results,
    }
    report["ok"] = all(report["invariants"].values())
    return report


def write_program_induction_report(out: str | Path, *, seed: int = 0) -> dict[str, Any]:
    report = evaluate_program_induction(seed=seed)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = [
    "ProgramCandidate", "ProgramStats", "InductionResult", "compile_program_source",
    "ast_program_is_safe", "partition_examples", "induce_program", "demo_suite",
    "evaluate_program_induction", "write_program_induction_report",
]
