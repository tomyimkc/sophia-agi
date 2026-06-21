"""Verifier synthesis — write checks for tasks no one hand-coded, then trust them
only as far as they are *measured*.

This is the bridge from "trustworthy narrow loop" toward generality: the loop is
only as broad as its verifiers, and we cannot hand-write a verifier for a task we
have never seen. So we synthesise one — and, crucially, we refuse to trust it
until it has been validated against independent ground truth.

Three steps; the third is what earns the claim:

  1. SYNTHESISE  — from a few oracle-labelled ``(answer, correct?)`` examples of a
     novel task, *fit* a library of parameterised check templates, producing
     candidate verifiers. No task-specific code was written in advance.
  2. META-VERIFY — *verify the verifier*: score each candidate on a HELD-OUT
     validation split whose labels come from an independent oracle (never from the
     candidate). Admit only candidates whose measured precision AND recall clear a
     floor. An unvalidated synthesised check is a hypothesis, not a gate.
  3. ABSTAIN     — if nothing clears the floor, synthesise NOTHING and declare the
     task unverifiable, rather than shipping a plausible-looking wrong check.
     (Competence where no verifier exists then lives in :mod:`agent.calibration`.)

Why this is not circular (enforced here, asserted in tests):
  - labels come from a caller-supplied oracle, never from the candidates;
  - fit / validate / test splits are disjoint (asserted), partitioned by a stable
    per-example index so the test split is never seen during synthesis;
  - the reported number is the TEST precision/recall of the admitted gate, plus an
    ablation: WITHOUT meta-verification, abstention on unverifiable tasks collapses
    (false-admission ~100%) and in-library precision degrades — so the
    *meta-verification*, not the template library, is what earns the trust.

Deterministic (no model call) so the mechanism is isolated and reproducible. A
model-proposes-a-predicate variant slots into ``TEMPLATES`` without touching the
meta-verification contract: a model only *widens* candidate generation; trust
still comes solely from measured validation.
"""

from __future__ import annotations

import ast
import math
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.verifiers import _fail, _ok

Predicate = Callable[[Any], bool]


# --------------------------------------------------------------------------- #
# A synthesised candidate check.
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    """A fitted check: ``predicate(answer) -> bool`` (True = looks acceptable)."""

    name: str
    params: dict
    predicate: Predicate

    def __call__(self, answer: Any) -> bool:
        try:
            return bool(self.predicate(answer))
        except Exception:
            return False


@dataclass
class VerifierStats:
    """How well a candidate classifies *correctness* on a labelled split.

    The positive class is the *incorrect* answer — the thing a gate must catch.
    ``precision`` = of answers the candidate rejects, the fraction truly
    incorrect (high precision ⇒ it rarely rejects a good answer). ``recall`` = of
    incorrect answers, the fraction the candidate rejects.
    """

    name: str
    precision: float
    recall: float
    accuracy: float
    n: int
    flagged: int


def _num(x: Any) -> "float | None":
    try:
        if isinstance(x, bool):
            return None
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


# --------------------------------------------------------------------------- #
# Template library. Each fitter inspects the labelled FIT examples and returns
# zero or more candidate checks. None of them know the task's hidden rule; they
# only generalise patterns shared by the correct answers.
# --------------------------------------------------------------------------- #


def _fit_equals_oracle(corrects, incorrects, task) -> list:
    """Strongest check: the answer must equal an independent recomputation.
    Only available when the task carries a callable ``oracle(task) -> expected``.
    """
    oracle = task.get("oracle")
    if not callable(oracle):
        return []
    try:
        expected = oracle(task)
    except Exception:
        return []

    def pred(a, e=expected):
        na, ne = _num(a), _num(e)
        if na is not None and ne is not None:
            return abs(na - ne) <= 1e-6
        return str(a).strip() == str(e).strip()

    return [Candidate("equals_oracle", {"expected": expected}, pred)]


def _fit_range(corrects, incorrects, task) -> list:
    nums = [_num(a) for a in corrects]
    if not nums or any(n is None for n in nums):
        return []
    lo, hi = min(nums), max(nums)
    return [Candidate("in_range", {"lo": lo, "hi": hi},
                      lambda a, lo=lo, hi=hi: (_num(a) is not None and lo <= _num(a) <= hi))]


def _fit_integer(corrects, incorrects, task) -> list:
    nums = [_num(a) for a in corrects]
    if not nums or any(n is None for n in nums):
        return []
    if all(float(n).is_integer() for n in nums):
        return [Candidate("is_integer", {},
                          lambda a: (_num(a) is not None and float(_num(a)).is_integer()))]
    return []


def _fit_sign(corrects, incorrects, task) -> list:
    nums = [_num(a) for a in corrects]
    if not nums or any(n is None for n in nums):
        return []
    out = []
    if all(n > 0 for n in nums):
        out.append(Candidate("positive", {}, lambda a: _num(a) is not None and _num(a) > 0))
    if all(n >= 0 for n in nums) and not all(n > 0 for n in nums):
        out.append(Candidate("nonneg", {}, lambda a: _num(a) is not None and _num(a) >= 0))
    return out


def _fit_divisible(corrects, incorrects, task) -> list:
    nums = [_num(a) for a in corrects]
    if not nums or any(n is None or not float(n).is_integer() for n in nums):
        return []
    out = []
    for k in (2, 3, 5, 7, 10):
        if all(int(n) % k == 0 for n in nums):
            out.append(Candidate(f"divisible_by_{k}", {"k": k},
                       lambda a, k=k: (_num(a) is not None and float(_num(a)).is_integer()
                                       and int(_num(a)) % k == 0)))
    return out


def _fit_prime(corrects, incorrects, task) -> list:
    nums = [_num(a) for a in corrects]
    if not nums or any(n is None or not float(n).is_integer() for n in nums):
        return []
    if all(_is_prime(int(n)) for n in nums):
        return [Candidate("is_prime", {},
                          lambda a: (_num(a) is not None and float(_num(a)).is_integer()
                                     and _is_prime(int(_num(a)))))]
    return []


_REGEX_CATALOG = {
    "digits_only": r"^-?\d+$",
    "decimal": r"^-?\d+\.\d+$",
    "has_letter": r"[A-Za-z]",
    "email_like": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    "iso_date": r"^\d{4}-\d{2}-\d{2}$",
    "uppercase": r"^[A-Z][A-Z\s]*$",
    "alnum_token": r"^[A-Za-z0-9_]+$",
}


def _fit_regex(corrects, incorrects, task) -> list:
    out = []
    for name, pat in _REGEX_CATALOG.items():
        rx = re.compile(pat)
        if corrects and all(rx.search(str(a)) for a in corrects):
            out.append(Candidate(f"regex:{name}", {"pattern": pat},
                       lambda a, rx=rx: bool(rx.search(str(a)))))
    return out


def _fit_length(corrects, incorrects, task) -> list:
    lens = [len(str(a)) for a in corrects]
    if not lens:
        return []
    lo, hi = min(lens), max(lens)
    return [Candidate("length_range", {"lo": lo, "hi": hi},
            lambda a, lo=lo, hi=hi: lo <= len(str(a)) <= hi)]


def _fit_contains(corrects, incorrects, task) -> list:
    # Only meaningful for non-numeric answers; let other templates own numbers.
    if not corrects or any(_num(a) is not None for a in corrects):
        return []
    word_sets = [set(re.findall(r"[a-z0-9]+", str(a).lower())) for a in corrects]
    common = sorted(w for w in set.intersection(*word_sets) if len(w) >= 3) if word_sets else []
    return [Candidate(f"contains:{w}", {"token": w}, lambda a, w=w: w in str(a).lower())
            for w in common[:3]]


# Order matters only for tie-breaking display; admission is by measured stats.
TEMPLATES: list = [
    _fit_equals_oracle, _fit_range, _fit_integer, _fit_sign, _fit_divisible,
    _fit_prime, _fit_regex, _fit_length, _fit_contains,
]


# --------------------------------------------------------------------------- #
# Model proposer — a model WIDENS candidate generation by writing predicates the
# template library cannot express. It never confers trust: a proposed predicate
# is a candidate like any other and must clear the SAME meta-verification floor.
# --------------------------------------------------------------------------- #

# Builtins a proposed ``check(answer)`` may use — pure SCALAR ops only. No
# range/list/set/sorted/map/filter (allocation), no getattr/type/__import__
# (traversal): a proposed predicate classifies one answer value, nothing more.
_SAFE_BUILTINS = {
    "len": len, "str": str, "int": int, "float": float, "abs": abs, "bool": bool,
    "min": min, "max": max, "round": round,
}
_ALLOWED_CALLS = set(_SAFE_BUILTINS)
_MAX_PREDICATE_SOURCE = 2000

# AST allowlist for a proposed predicate. This REPLACES a substring blocklist
# (which a runtime-built dunder + ``str.format`` could bypass, per adversarial
# review). Only scalar arithmetic / comparison / boolean logic and calls to the
# safe builtins are permitted. Everything else is rejected by construction:
#   - ast.Attribute absent  -> no dunder traversal ('().__class__...'), no '.format'
#   - For/While/comprehensions absent -> no unbounded CPU (infinite loop)
#   - range/containers/Mult(*)/Pow(**) absent -> no allocation bomb
#   - Import/Lambda absent -> no code loading / indirection
_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Expr,
    ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.IfExp, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mod, ast.FloorDiv, ast.Div,        # NOT Mult / Pow
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)

ProposeFn = Callable[[dict, list, list], list]   # (task, corrects, incorrects) -> [source]


def _ast_predicate_is_safe(tree: ast.AST) -> bool:
    """Pass iff the tree is a single ``def check(...)`` using only allowlisted
    nodes and calling only the safe builtins (no attribute access, loops,
    comprehensions, imports, lambdas, multiplication or power)."""
    body = getattr(tree, "body", [])
    funcs = [n for n in body if isinstance(n, ast.FunctionDef)]
    if len(body) != 1 or not funcs or funcs[0].name != "check" or funcs[0].decorator_list:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS
        ):
            return False
    return True


def _compile_predicate(src: str) -> "Predicate | None":
    """Compile a model-proposed ``def check(answer): ...`` under an AST allowlist
    and restricted builtins. Returns the callable, or None if unsafe/malformed."""
    if not isinstance(src, str) or "check" not in src or len(src) > _MAX_PREDICATE_SOURCE:
        return None
    try:
        tree = ast.parse(src, "<proposed-predicate>", "exec")
    except SyntaxError:
        return None
    if not _ast_predicate_is_safe(tree):
        return None
    ns: dict = {}
    try:
        exec(compile(tree, "<proposed-predicate>", "exec"),   # noqa: S102 — AST-allowlisted, restricted env
             {"__builtins__": _SAFE_BUILTINS}, ns)
    except Exception:
        return None
    fn = ns.get("check")
    return fn if callable(fn) else None


def propose_predicates(task: dict, corrects: list, incorrects: list, *,
                       propose_fn: ProposeFn, max_candidates: int = 8) -> list:
    """Turn model-proposed predicate sources into Candidates (sandbox-compiled).

    ``propose_fn(task, corrects, incorrects)`` returns a list of Python sources,
    each defining ``def check(answer): -> bool``. Unsafe/malformed sources are
    dropped; the survivors are returned as Candidates to be meta-verified exactly
    like template-fitted ones. Disabled via ``$SOPHIA_ALLOW_PROPOSED_PREDICATES=0``.
    """
    if os.environ.get("SOPHIA_ALLOW_PROPOSED_PREDICATES", "1").strip() in ("0", "false", "no"):
        return []
    try:
        sources = list(propose_fn(task, corrects, incorrects) or [])
    except Exception:
        return []
    out: list = []
    for i, src in enumerate(sources[:max_candidates]):
        fn = _compile_predicate(src)
        if fn is None:
            continue
        out.append(Candidate(f"proposed:{i}", {"src": src}, (lambda f: lambda a: bool(f(a)))(fn)))
    return out


# --------------------------------------------------------------------------- #
# Meta-verification: verify the verifier against an independent oracle's labels.
# --------------------------------------------------------------------------- #


def score_predicate(predicate: Predicate, examples: list) -> VerifierStats:
    """Score a predicate as a *correctness classifier* over labelled examples.

    ``examples`` are ``{"answer": ..., "label": bool}`` where ``label`` is the
    INDEPENDENT oracle's verdict (True = the answer is correct). The predicate is
    a candidate verifier; rejecting (predicate False) an incorrect answer is a
    true positive.
    """
    tp = fp = fn = tn = 0
    for ex in examples:
        flagged = not bool(predicate(ex["answer"]))      # the verifier rejects it
        incorrect = not bool(ex["label"])                # oracle says it is wrong
        if incorrect and flagged:
            tp += 1
        elif incorrect and not flagged:
            fn += 1
        elif (not incorrect) and flagged:
            fp += 1
        else:
            tn += 1
    n = len(examples)
    precision = tp / (tp + fp) if (tp + fp) else 1.0     # rejects nothing ⇒ vacuously precise
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    return VerifierStats(
        name=getattr(predicate, "name", "predicate"),
        precision=round(precision, 4), recall=round(recall, 4),
        accuracy=round(accuracy, 4), n=n, flagged=tp + fp,
    )


@dataclass
class SynthesisResult:
    task_id: str
    admitted: list = field(default_factory=list)      # [(Candidate, VerifierStats)]
    rejected: list = field(default_factory=list)      # [(Candidate, VerifierStats)]
    gate: "Callable | None" = None                    # composed Verifier, or None if abstained
    test_stats: "VerifierStats | None" = None
    abstained: bool = True
    splits: dict = field(default_factory=dict)
    meta_verified: bool = True

    def report(self) -> dict:
        return {
            "taskId": self.task_id,
            "abstained": self.abstained,
            "metaVerified": self.meta_verified,
            "admitted": [{"name": c.name, "precision": s.precision, "recall": s.recall} for c, s in self.admitted],
            "rejectedCount": len(self.rejected),
            "splits": self.splits,
            "testStats": None if self.test_stats is None else {
                "precision": self.test_stats.precision,
                "recall": self.test_stats.recall,
                "accuracy": self.test_stats.accuracy,
                "n": self.test_stats.n,
            },
        }


def _partition(examples: list, fit_frac: float, val_frac: float, seed: int):
    """Stable, disjoint fit/val/test partition. The test slice is never touched
    during synthesis — that disjointness is the non-circularity guarantee."""
    indexed = [dict(ex, _idx=ex.get("_idx", i)) for i, ex in enumerate(examples)]
    rng = random.Random(seed)
    rng.shuffle(indexed)
    n = len(indexed)
    n_fit = max(1, int(round(n * fit_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_fit = min(n_fit, max(1, n - 2))                    # always leave room for val+test
    n_val = min(n_val, max(1, n - n_fit - 1))
    fit = indexed[:n_fit]
    val = indexed[n_fit:n_fit + n_val]
    test = indexed[n_fit + n_val:]
    return fit, val, test


def compose(admitted: list, mode: str = "all") -> "Callable[[Any], bool]":
    """Combine admitted candidates into one predicate. ``all`` (default,
    conjunction) rejects an answer if ANY admitted check rejects it — maximal
    error-catching; ``any`` rejects only when ALL reject. The composition is held
    fixed across the WITH/WITHOUT-meta-verification ablation so the only variable
    is whether candidates were validated before admission."""
    preds = [c for c, _ in admitted]
    if not preds:
        return lambda a: True
    if mode == "any":
        return lambda a: any(p(a) for p in preds)
    return lambda a: all(p(a) for p in preds)


def synthesize(
    task: dict, *,
    fit_frac: float = 0.4, val_frac: float = 0.3,
    min_precision: float = 0.95, min_recall: float = 0.8,
    seed: int = 0, meta_verify: bool = True, compose_mode: str = "all",
    propose_fn: "ProposeFn | None" = None,
) -> SynthesisResult:
    """Synthesise a verifier for ``task`` from its oracle-labelled ``examples``.

    With ``meta_verify`` (default) a candidate is admitted only if its precision
    AND recall on the disjoint VALIDATION split clear the floors. With
    ``meta_verify=False`` (the ablation) every fitted candidate is admitted
    unchecked — modelling "trust the synthesised check without verifying it".

    ``propose_fn`` (optional) lets a model widen candidate generation with
    predicates the template library cannot express; proposed candidates clear the
    SAME meta-verification floor, so trust still comes only from measured validation.
    """
    examples = list(task.get("examples", []))
    res = SynthesisResult(task_id=str(task.get("task_id", "?")), meta_verified=meta_verify)
    if len(examples) < 4:
        res.splits = {"fit": 0, "val": 0, "test": 0, "note": "too few examples"}
        return res

    fit, val, test = _partition(examples, fit_frac, val_frac, seed)
    res.splits = {"fit": len(fit), "val": len(val), "test": len(test)}

    corrects = [e["answer"] for e in fit if e["label"]]
    incorrects = [e["answer"] for e in fit if not e["label"]]
    if not corrects:
        return res                                       # nothing to generalise from → abstain

    candidates: list = []
    for tmpl in TEMPLATES:
        try:
            candidates.extend(tmpl(corrects, incorrects, task))
        except Exception:
            continue
    if propose_fn is not None:
        candidates.extend(propose_predicates(task, corrects, incorrects, propose_fn=propose_fn))

    judge_split = val or fit
    for cand in candidates:
        stats = score_predicate(cand, judge_split)
        stats.name = cand.name
        if (not meta_verify) or (stats.precision >= min_precision and stats.recall >= min_recall):
            res.admitted.append((cand, stats))
        else:
            res.rejected.append((cand, stats))

    if not res.admitted:
        return res                                       # ABSTAIN: nothing trustworthy

    res.abstained = False
    gate_pred = compose(res.admitted, mode=compose_mode)
    res.gate = as_verifier(gate_pred, name="+".join(c.name for c, _ in res.admitted))
    if test:
        res.test_stats = score_predicate(gate_pred, test)
    return res


def as_verifier(predicate: Predicate, name: str = "synthesized") -> Callable:
    """Adapt a predicate over an *answer* into the harness ``Verifier`` signature
    ``(text, task, step) -> {passed, reasons, detail}`` so a synthesised gate
    drops straight into :mod:`agent.harness` / :mod:`agent.guarded`."""

    def _verify(text: str, task: Any = None, step: dict | None = None) -> dict:
        if bool(predicate(text)):
            return _ok({"check": name})
        return _fail([f"failed synthesized check: {name}"], {"check": name})

    return _verify


# --------------------------------------------------------------------------- #
# Aggregate evaluation across many tasks — the falsifiable headline + ablation.
# --------------------------------------------------------------------------- #


def evaluate_suite(
    tasks: list, *, in_library_ids: "set | None" = None,
    out_of_library_ids: "set | None" = None, seed: int = 0,
    meta_verify: bool = True, **kw,
) -> dict:
    """Run synthesis over a suite and summarise the numbers that matter.

    Reports, separately for tasks whose true rule the template library CAN and
    CANNOT express:
      - mean TEST precision/recall of the synthesised gate (capability), and
      - abstention rate (safety — on out-of-library tasks, abstaining is correct).
    """
    in_ids = in_library_ids or set()
    out_ids = out_of_library_ids or set()
    per_task = []
    for t in tasks:
        r = synthesize(t, seed=seed, meta_verify=meta_verify, **kw)
        per_task.append((t, r))

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    in_results = [r for t, r in per_task if str(t.get("task_id")) in in_ids]
    out_results = [r for t, r in per_task if str(t.get("task_id")) in out_ids]

    in_prec = [r.test_stats.precision for r in in_results if r.test_stats]
    in_rec = [r.test_stats.recall for r in in_results if r.test_stats]
    # Any out-of-library task that did NOT abstain is a false admission; surface
    # its TEST precision/recall so a plausible-looking wrong gate cannot hide
    # behind a bare abstention count (see the safety invariants in run_demo).
    out_admitted = [
        {"taskId": str(t.get("task_id")),
         "testPrecision": (r.test_stats.precision if r.test_stats else None),
         "testRecall": (r.test_stats.recall if r.test_stats else None)}
        for t, r in per_task
        if str(t.get("task_id")) in out_ids and not r.abstained
    ]
    return {
        "metaVerified": meta_verify,
        "nTasks": len(tasks),
        "inLibrary": {
            "n": len(in_results),
            "meanTestPrecision": _mean(in_prec),
            "meanTestRecall": _mean(in_rec),
            "abstentionRate": round(sum(r.abstained for r in in_results) / len(in_results), 4) if in_results else 0.0,
        },
        "outOfLibrary": {
            "n": len(out_results),
            # On out-of-library tasks the CORRECT behaviour is to abstain.
            "abstentionRate": round(sum(r.abstained for r in out_results) / len(out_results), 4) if out_results else 0.0,
            "falseAdmissionRate": round(sum(not r.abstained for r in out_results) / len(out_results), 4) if out_results else 0.0,
            "admittedGates": out_admitted,
            "worstAdmittedTestPrecision": max((g["testPrecision"] or 0.0 for g in out_admitted), default=0.0),
        },
        "reports": [r.report() for _, r in per_task],
    }
