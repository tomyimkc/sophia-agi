# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Executor tools for the rollout factory (safe, dependency-free).

A tool is ``Callable[[str], str]`` — it takes the raw argument string from a
``TOOL: name(arg)`` call the executor emits and returns an observation string that
is appended back into the executor session. Every tool turn deepens the session,
which is precisely what makes the append-only prefix cache pay off (a single-shot
answer never accumulates a prefix worth caching).

``safe_calc`` is an AST-restricted arithmetic evaluator — numbers and ``+ - * / **
%`` with parentheses and unary signs only. No names, calls, attributes, or
indexing, so there is no ``eval`` surface. This is the natural physics/math tool:
the model proposes the formula and substitution; the tool does the ground-truth
arithmetic.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Callable

Tool = Callable[[str], str]

# A tool call the executor can emit, e.g. ``TOOL: calc(2*3 + 4)``.
TOOL_RE = re.compile(r"TOOL:\s*([A-Za-z_]\w*)\(([^)]*)\)")

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def safe_calc(arg: str) -> str:
    """Evaluate a pure-arithmetic expression. Returns ``= <value>`` or ``error: …``.

    Restricted to numeric literals and ``+ - * / ** %`` — no names or calls, so it
    is safe to run on model-generated strings.
    """
    expr = (arg or "").strip()
    if not expr or len(expr) > 256:
        return "error: empty or oversized expression"
    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_node(tree)
    except Exception as exc:  # noqa: BLE001 - any malformed/forbidden expr -> error obs
        return f"error: {exc}"
    # Render integers cleanly, floats with reasonable precision.
    if value == int(value):
        return f"= {int(value)}"
    return f"= {value:g}"


DEFAULT_TOOLS: "dict[str, Tool]" = {"calc": safe_calc}
