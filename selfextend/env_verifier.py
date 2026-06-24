# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Environment-as-verifier: verify by EXECUTING the candidate, not by judging it.

The strongest verification needs no LLM judge — the environment decides. Two safe,
deterministic backends (no arbitrary code execution):

  - ``arithmetic``: evaluate the candidate expression with an AST-restricted numeric
    evaluator and compare to the expected value.
  - ``regex``:      run the candidate pattern against labelled cases; it passes iff
    every case's match/no-match matches its label.

This is the same principle as the repo's interpreter-as-verifier for code, generalized
to a tiny multi-domain action surface — and the reward signal a verifiable-action loop
would optimize.
"""

from __future__ import annotations

import ast
import operator
import re

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv,
}


def safe_arith(expr: str) -> float:
    """Evaluate a pure-arithmetic expression. Raises ValueError on anything else."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


def verify_by_execution(kind: str, candidate: str, spec: dict) -> dict:
    """Return {passed, detail}. Fail-closed: any execution error => passed False."""
    try:
        if kind == "arithmetic":
            got = safe_arith(candidate)
            ok = abs(got - float(spec["expected"])) < 1e-9
            return {"passed": ok, "detail": {"got": got, "expected": spec["expected"]}}
        if kind == "regex":
            pat = re.compile(candidate, re.IGNORECASE)
            results = [(bool(pat.search(text)) == bool(should)) for text, should in spec["cases"]]
            return {"passed": all(results), "detail": {"perCase": results}}
        return {"passed": False, "detail": f"unknown kind {kind!r}"}
    except (ValueError, re.error, SyntaxError, KeyError, TypeError) as exc:
        return {"passed": False, "detail": repr(exc)}
